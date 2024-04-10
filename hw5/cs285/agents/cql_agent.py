from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu
from cs285.agents.dqn_agent import DQNAgent


class CQLAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        cql_alpha: float,
        cql_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.cql_alpha = cql_alpha
        self.cql_temperature = cql_temperature

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: bool,
    ) -> Tuple[torch.Tensor, dict, dict]:
        loss, metrics, variables = super().compute_critic_loss(
            obs,
            action,
            reward,
            next_obs,
            done,
        )

        # TODO(student): modify the loss to implement CQL
        # Hint: `variables` includes qa_values and q_values from your CQL implementation
        
        # Compute the log-sum-exp of the Q-values for all actions, scaled by temperature
        # q-values = q-values that were actually taken!
        # qa_values.shape == (batch_size, self.num_actions)
        logsumexp_q_values = torch.logsumexp(variables['qa_values'] / self.cql_temperature, dim=1, keepdim=False) # TODO Musti: Double check cql_temperature term!!

         # Compute the CQL regularizer term
        cql_regularizer = self.cql_alpha * (logsumexp_q_values - variables['q_values']).mean()

        # Add the CQL regularizer to the original loss
        loss += cql_regularizer

        return loss, metrics, variables
