import torch as th
import torch.nn as nn
from torch import Tensor
from rl_mind.nn import build_mlp
from rl_mind.core import Action, Actor


def mlp(sizes: list[int], layer_norm: bool = False) -> nn.Sequential:
    """Like `build_mlp` (ReLU activations), with an optional LayerNorm after each hidden linear layer

    `Linear -> LayerNorm -> ReLU` for the hidden layers, and a plain `Linear` output layer.
    Without LayerNorm, the linear layers (and their initialization) are the same as `build_mlp`.
    """
    layers: list[nn.Module] = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(n_in, n_out))
        if i < len(sizes) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(n_out))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ContinuousQNetwork(nn.Module):
    """The Q-network $Q(s, a)$ for continuous actions"""
    def __init__(self,
                obs_dim: int,
                hidden: tuple[int,...],
                action_dim: int,
                layer_norm: bool = False):
        super().__init__()
        self.model = mlp([obs_dim + action_dim, *hidden, 1], layer_norm=layer_norm)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """Compute $Q(s, a)$ for a batch: `[B, obs_dim] x [B, action_dim] -> [B]`"""
        return self.model(th.cat([obs, action], dim=1)).squeeze(-1)


class ContinousDeterministicActor(Actor[Action]):
    """Creates a new actor with continuous actions in `action_dim` dimensions."""
    def __init__(self,
                 obs_dim: int,
                 hidden: tuple[int,...],
                 action_dim: int):
          super().__init__()
          self.model = build_mlp([obs_dim, *hidden, action_dim], output_activation=nn.Tanh())

    def forward(self, obs: Tensor) -> Action:
        return Action(value=self.model(obs))


class GaussianNoise(Actor[Action]):
    """Adds Gaussian noise to the actions of another actor (at training time)"""
    def __init__(self,
                 actor: Actor[Action],
                 sigma: float):
        super().__init__()
        self.actor = actor
        self.sigma = sigma

    def forward(self, obs):
        action = self.actor(obs).value
        # Clip to the action space so that the replay buffer only stores valid actions
        return Action(value=(action + self.sigma * th.randn_like(action)).clamp(-1, 1))

    def act(self, obs):
        return self.actor.act(obs)

