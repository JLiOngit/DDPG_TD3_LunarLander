import torch as th
from torch import Tensor
from rl_mind.core import Action
from rl_mind.data import Transitions


def compute_critic_loss(gamma: float, batch: Transitions[Action], q_values: Tensor, next_q_values: Tensor) -> Tensor:
    """Compute the DDPG critic loss from a batch of transition"""
    target = batch.reward + gamma * (1-batch.terminated.float()) * next_q_values
    critic_loss = th.sum(th.nn.functional.mse_loss(q_values, target))
    return critic_loss


def compute_actor_loss(q_values: Tensor) -> Tensor:
        """Return the actor loss given $Q(s_t, \\pi(s_t))$ (shape `[B]`)"""
        return -th.mean(q_values)

