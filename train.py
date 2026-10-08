"""Training loops of DDPG and TD3, with the Monte Carlo estimate of the overestimation bias

The functions live in a module (and not in the notebook) so that `run_one` can be
run in separate processes.
"""

import copy
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from rl_mind.nn import soft_update
from rl_mind.env import VecEnv
from rl_mind.data import ReplayBuffer
from rl_mind.collectors import TransitionCollector
from rl_mind.evaluation import EvalResult, Evaluator
from rl_mind.notebook import run_directory, silence_known_warnings

from agent import *
from loss import *
from config import *
from bias import *
from results import save_results


def run_ddpg(cfg: DDPGConfig) -> tuple[Evaluator, list[tuple[int, dict[str, float]]]]:
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed)
    # Create the actor, the critic and its target, and the optimizers
    actor = ContinousDeterministicActor(obs_dim=env.observation_dim, hidden=cfg.actor_hidden, action_dim=env.action_dim)
    critic = ContinuousQNetwork(obs_dim=env.observation_dim, hidden=cfg.critic_hidden, action_dim=env.action_dim, layer_norm=cfg.layer_norm)
    # the target critic is a copy of the critic (`copy.deepcopy`)
    target_critic = copy.deepcopy(critic)
    # the target actor $\mu'$ is a copy of the actor
    target_actor = copy.deepcopy(actor)
    # one optimizer per network (`cfg.lr_actor`, `cfg.lr_critic`)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.lr_actor)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=cfg.lr_critic)
    # Data collection (with exploration noise), replay buffer and evaluation
    collector = TransitionCollector(env, GaussianNoise(actor, cfg.action_noise))
    buffer = ReplayBuffer(cfg.buffer_size)
    name = run_name("ddpg", cfg)
    run_dir = run_directory(name)
    # Evaluation episodes (deterministic actor), also used to measure the overestimation bias
    evaluator = BiasEvaluator(
        VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100),
        every=cfg.eval_interval,
        critic=critic,
        gamma=cfg.gamma,
        horizon=cfg.mc_horizon,
        run_dir=run_dir,
        writer=SummaryWriter(run_dir),
    )
    bias_history = []
    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        buffer.add(collector.collect(cfg.steps_per_update))
        pbar.update(collector.steps - pbar.n)
        if len(buffer) < cfg.learning_starts:
            continue
        for _ in range(cfg.utd_ratio):
            batch = buffer.sample(cfg.batch_size)
            # Q-values of the actions that were played (this is where gradients flow)
            q_values = critic(batch.obs, batch.action.value)
            with torch.no_grad():
                # Q-values of the *target* actor's actions in the next states, from the target critic
                next_action = target_actor(batch.next_obs)
                next_q_values = target_critic(batch.next_obs, next_action.value)
            # Critic update: compute the loss, then a gradient step
            critic_loss = compute_critic_loss(cfg.gamma, batch, q_values, next_q_values)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            # Actor update: maximize Q(s, pi(s)), with a gradient step on the actor only
            actor_loss = compute_actor_loss(critic(batch.obs, actor(batch.obs).value))
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            # Soft update of the target networks (`soft_update`)
            soft_update(actor, target_actor, cfg.tau)
            soft_update(critic, target_critic, cfg.tau)
        # Losses of the last gradient step
        evaluator.writer.add_scalar("loss/critic", critic_loss.item(), collector.steps)
        evaluator.writer.add_scalar("loss/actor", actor_loss.item(), collector.steps)
        if result := evaluator.run_if_needed(collector.steps, actor):
            # Networks are frozen here: Q(s_t, a_t) vs the Monte Carlo returns G_t of the evaluation episodes
            bias = evaluator.last_metrics
            bias_history.append((collector.steps, bias))
            save_results(run_dir, name, cfg, evaluator.history, bias_history)
            for key, value in bias.items():
                evaluator.writer.add_scalar(f"bias/{key}", value, collector.steps)
            pbar.set_description(
                f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f}"
            )
    pbar.close()
    save_results(run_dir, name, cfg, evaluator.history, bias_history, done=True)
    return evaluator, bias_history


def run_td3(cfg: TD3Config) -> tuple[Evaluator, list[tuple[int, dict[str, float]]]]:
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed)
    # Create the actor, the two critics, their three targets and the optimizers
    actor = ContinousDeterministicActor(env.observation_dim, cfg.actor_hidden, env.action_dim)
    target_actor = copy.deepcopy(actor)
    critic_1 = ContinuousQNetwork(env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.layer_norm)
    critic_2 = ContinuousQNetwork(env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.layer_norm)
    target_critic_1 = copy.deepcopy(critic_1)
    target_critic_2 = copy.deepcopy(critic_2)
    # one optimizer for the actor, one for both critics (give it the
    # parameters of both)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.lr_actor)
    critic_optimizer = torch.optim.AdamW(list(critic_1.parameters()) + list(critic_2.parameters()), lr=cfg.lr_critic)
    # Data collection (with exploration noise), replay buffer and evaluation
    collector = TransitionCollector(env, GaussianNoise(actor, cfg.action_noise))
    buffer = ReplayBuffer(cfg.buffer_size)
    name = run_name("td3", cfg)
    run_dir = run_directory(name)
    # Evaluation episodes (deterministic actor), also used to measure the overestimation bias
    evaluator = BiasEvaluator(
        VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100),
        every=cfg.eval_interval,
        critic=critic_1,
        gamma=cfg.gamma,
        horizon=cfg.mc_horizon,
        run_dir=run_dir,
        writer=SummaryWriter(run_dir),
    )
    updates = 0  # number of gradient steps (for the policy delay)
    bias_history = []
    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        buffer.add(collector.collect(cfg.steps_per_update))
        pbar.update(collector.steps - pbar.n)
        if len(buffer) < cfg.learning_starts:
            continue
        for _ in range(cfg.utd_ratio):
            batch = buffer.sample(cfg.batch_size)
            with torch.no_grad():
                # Target policy smoothing: target actor + clipped Gaussian noise, clipped to [-1, 1]
                target_actions = target_actor(batch.next_obs).value
                noise = (torch.randn_like(target_actions) * cfg.target_noise).clamp(
                    -cfg.target_noise_clip, cfg.target_noise_clip
                )
                next_actions = (target_actions + noise).clamp(-1, 1)
                # Clipped double-Q: a single target, from the min of the two target critics
                next_q_values = torch.min(target_critic_1(batch.next_obs, next_actions),
                                     target_critic_2(batch.next_obs, next_actions))
            # Critic update: both critics are regressed towards the same target
            q_values_1 = critic_1(batch.obs, batch.action.value)
            q_values_2 = critic_2(batch.obs, batch.action.value)
            critic_loss = (compute_critic_loss(cfg.gamma, batch, q_values_1, next_q_values)
                       + compute_critic_loss(cfg.gamma, batch, q_values_2, next_q_values))
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            updates += 1
            if updates % cfg.policy_delay == 0:
                # Delayed policy update: update the actor (with `critic_1`)
                actor_loss = compute_actor_loss(critic_1(batch.obs, actor(batch.obs).value))
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                evaluator.writer.add_scalar(
                    "loss/actor", actor_loss.item(), collector.steps
                )
                # Soft update of the three target networks
                soft_update(actor, target_actor, cfg.tau)
                soft_update(critic_1, target_critic_1, cfg.tau)
                soft_update(critic_2, target_critic_2, cfg.tau)
        evaluator.writer.add_scalar("loss/critic", critic_loss.item(), collector.steps)
        if result := evaluator.run_if_needed(collector.steps, actor):
            # Networks are frozen here: Q(s_t, a_t) vs the Monte Carlo returns G_t of the evaluation episodes
            bias = evaluator.last_metrics
            bias_history.append((collector.steps, bias))
            save_results(run_dir, name, cfg, evaluator.history, bias_history)
            for key, value in bias.items():
                evaluator.writer.add_scalar(f"bias/{key}", value, collector.steps)
            pbar.set_description(
                f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f}"
            )
    pbar.close()
    save_results(run_dir, name, cfg, evaluator.history, bias_history, done=True)
    return evaluator, bias_history



def run_one(algo: str, seed: int, **overrides) -> tuple[list[EvalResult], list[tuple[int, dict[str, float]]]]:
    """One run, meant to be executed in its own process

    Returns only what the analysis needs (an `Evaluator` cannot be sent back
    to the notebook: it holds a tensorboard writer and environments).

    :param algo: `"ddpg"` or `"td3"`
    :param seed: The random seed
    :param overrides: Fields of the configuration to change, e.g. `max_steps=30_000`
    """
    torch.set_num_threads(1)  # one core per process, so that runs do not compete
    silence_known_warnings()
    # The progress bars of parallel runs would be interleaved in the notebook:
    # follow the runs in tensorboard instead (errors are still sent back as exceptions)
    sys.stderr = open(os.devnull, "w")
    config, run = {"ddpg": (DDPGConfig, run_ddpg), "td3": (TD3Config, run_td3)}[algo]
    cfg = replace(config(), seed=seed, **overrides)
    try:
        evaluator, bias_history = run(cfg)
    except Exception:
        # stderr is silenced: keep the error message in `outputs/errors/`
        errors = Path("outputs") / "errors"
        errors.mkdir(parents=True, exist_ok=True)
        (errors / f"{run_name(algo, cfg).replace('/', '_')}.txt").write_text(traceback.format_exc())
        raise
    return evaluator.history, bias_history


def run_job(job: tuple[str, int, dict]) -> tuple[list[EvalResult], list[tuple[int, dict[str, float]]]]:
    """`run_one` for a `(algo, seed, overrides)` job, e.g. `("td3", 1, {"layer_norm": True})`"""
    algo, seed, overrides = job
    return run_one(algo, seed, **overrides)
