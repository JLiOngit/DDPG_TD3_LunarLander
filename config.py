from dataclasses import dataclass, fields


@dataclass(frozen=True)
class DDPGConfig:
    env_name: str = "LunarLanderContinuous-v3"
    seed: int = 1
    #: Total number of environment steps
    max_steps: int = 300000
    #: Number of parallel training environments
    n_envs: int = 1
    #: Environment steps between two gradient updates
    steps_per_update: int = 1
    #: Steps before learning starts
    learning_starts: int = 1000
    #: Replay buffer capacity
    buffer_size: int = 300000
    batch_size: int = 128
    #: Discount factor
    gamma: float = 0.99
    #: Target network update coefficient
    tau: float = 0.005
    #: Exploration noise
    action_noise: float = 0.1
    actor_hidden: tuple[int, ...] = (256, 256)
    critic_hidden: tuple[int, ...] = (256, 256)
    #: LayerNorm after each hidden layer of the critic(s)
    layer_norm: bool = False
    lr_actor: float = 0.0001
    lr_critic: float = 0.001
    #: Steps between two evaluations, and number of evaluation episodes
    eval_interval: int = 10000
    n_eval_envs: int = 10
    #: Steps dropped at the end of truncated episodes for the bias estimate
    mc_horizon: int = 300
    utd_ratio: int = 1


@dataclass(frozen=True)
class TD3Config(DDPGConfig):
    #: Number of critic updates between two policy updates
    policy_delay: int = 2
    #: Std of the noise added to the target policy actions
    target_noise: float = 0.2
    #: Clipping of the target policy noise
    target_noise_clip: float = 0.5


def run_name(algo: str, cfg: DDPGConfig) -> str:
    """The name of a run, `<algo>/<variant>/S<seed>`

    The variant lists the fields that differ from the default configuration
    (e.g. `td3/layer_norm=True/S1`), or is `base` when there is none.
    """
    default = type(cfg)()
    changes = []
    for field in fields(cfg):
        value = getattr(cfg, field.name)
        if field.name == "seed" or value == getattr(default, field.name):
            continue
        if isinstance(value, tuple):
            value = "x".join(map(str, value))
        changes.append(f"{field.name}={value}")
    return f"{algo}/{'-'.join(changes) or 'base'}/S{cfg.seed}"
