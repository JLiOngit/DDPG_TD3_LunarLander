import torch as th
from rl_mind.core import Actor
from rl_mind.evaluation import Evaluator
from torch import Tensor


def discounted_returns(rewards: Tensor, gamma: float) -> Tensor:
    """Monte Carlo returns of one episode"""
    returns = th.zeros_like(rewards)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns


class BiasEvaluator(Evaluator):
    """An `Evaluator` that also measures the overestimation bias, on its own evaluation episodes

    Each evaluation plays exactly one episode per environment with the deterministic actor.
    On these same episodes, it computes the cumulated reward (the performance, as `Evaluator`),
    and compares $Q(s_t, a_t)$ to the Monte Carlo returns $G_t$ (the bias). The metrics of the
    last evaluation are in `last_metrics`.
    """
    def __init__(self, env, every: int, critic, gamma: float, horizon: int = 300, run_dir=None, writer=None):
        """
        :param critic: The critic whose bias is measured ($Q_1$ for TD3)
        :param horizon: Steps dropped at the end of truncated episodes, whose returns are cut short
        """
        super().__init__(env, every, run_dir=run_dir, writer=writer)
        self.critic = critic
        self.gamma = gamma
        self.horizon = horizon
        #: The metrics of the last evaluation (bias, episode lengths and outcomes)
        self.last_metrics: dict[str, float] = {}

    @th.no_grad()
    def run(self, actor: Actor) -> Tensor:
        n = self.env.num_envs
        obs = self.env.reset()
        cumulated = th.zeros(n)
        finished = th.zeros(n, dtype=th.bool)
        terminated = th.zeros(n, dtype=th.bool)
        lengths = th.zeros(n, dtype=th.long)
        last_reward = th.zeros(n)
        obs_list, action_list, reward_list = [], [], []
        while not bool(finished.all()):
            action = actor.act(obs)
            step = self.env.step(action)
            active = ~finished
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(step.reward)
            cumulated += step.reward * active
            lengths += active
            ended = step.done & active
            terminated |= step.terminated & active
            last_reward = th.where(ended, step.reward, last_reward)
            finished |= step.done
            obs = step.obs
        # [T_max, n, ...]: the episode of environment i is made of its first lengths[i] steps
        all_obs, all_actions, all_rewards = th.stack(obs_list), th.stack(action_list), th.stack(reward_list)
        q_list, g_list = [], []
        for i in range(n):
            length = int(lengths[i])
            returns = discounted_returns(all_rewards[:length, i], self.gamma)
            # Truncated episodes: the returns of the last `horizon` steps are cut short
            keep = length if terminated[i] else length - self.horizon
            if keep > 0:
                q_list.append(self.critic(all_obs[:keep, i], all_actions[:keep, i]))
                g_list.append(returns[:keep])
        # LunarLander ends with a reward of -100 (crash) or +100 (landed and at rest)
        self.last_metrics = {
            "episode_length": lengths.float().mean().item(),
            "landed": (terminated & (last_reward >= 100)).float().mean().item(),
            "crashed": (terminated & (last_reward <= -100)).float().mean().item(),
            "truncated": (~terminated).float().mean().item(),
        }
        if q_list:
            q, g = th.cat(q_list), th.cat(g_list)
            self.last_metrics |= {
                "bias": (q - g).mean().item(),
                "normalized_bias": ((q - g).mean() / g.abs().mean()).item(),
                "q": q.mean().item(),
                "mc_return": g.mean().item(),
            }
        return cumulated
