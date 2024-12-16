from copy import deepcopy
from typing import Any, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from gfn.actions import Actions
from gfn.containers import Trajectories
from gfn.env import Env
from gfn.modules import GFNModule
from gfn.preprocessors import Preprocessor
from gfn.samplers import Sampler
from gfn.states import States
from gfn.utils.distributions import UnsqueezedCategorical
from torch.distributions import Categorical, Distribution

from common.graph import Graph
from common.state_action import DistrictActions, DistrictState, stack_states_district


class DistrictPolicyEstimator(GFNModule):
    def __init__(
        self,
        module: nn.Module,
        n_actions: int,
        preprocessor: Preprocessor | None = None,
        is_backward: bool = False,
    ):
        """Initializes a estimator for P_F for discrete environments.

        Args:
            n_actions: Total number of actions in the Discrete Environment.
            is_backward: if False, then this is a forward policy, else backward policy.
        """
        super().__init__(module, preprocessor, is_backward=is_backward)
        self.n_actions = n_actions

    def expected_output_dim(self) -> int:
        if self.is_backward:
            return self.n_actions[0] - 1
        else:
            return self.n_actions[0]

    def to_probability_distribution(
        self,
        states: States,
        module_output: torch.Tensor,
        temperature: float = 1.0,
        sf_bias: float = 0.0,
        epsilon: float = 0.0,
    ) -> Categorical:
        """Returns a probability distribution given a batch of states and module output.

        We handle off-policyness using these kwargs.

        Args:
            states: The states to use.
            module_output: The output of the module as a tensor of shape (*batch_shape, output_dim).
            temperature: scalar to divide the logits by before softmax. Does nothing
                if set to 1.0 (default), in which case it's on policy.
            sf_bias: scalar to subtract from the exit action logit before dividing by
                temperature. Does nothing if set to 0.0 (default), in which case it's
                on policy.
            epsilon: with probability epsilon, a random action is chosen. Does nothing
                if set to 0.0 (default), in which case it's on policy."""
        self.check_output_dim(module_output)

        # masks = states.backward_masks if self.is_backward else states.forward_masks
        masks = torch.ones_like(module_output, dtype=torch.bool)
        logits = module_output
        logits[~masks] = -float("inf")

        #bkws shift
        shift = 1
        if self.is_backward:
            shift = 0
        logits_county = logits[:, :self.preprocessor.district_output_dim+shift]
        logits_district = logits[:, self.preprocessor.district_output_dim+shift:]
        # logits_county[:, -1] += 2.0
        masks_county = masks[:, :self.preprocessor.district_output_dim+shift]
        masks_district = masks[:, self.preprocessor.district_output_dim+shift:]

        # Forward policy supports exploration in many implementations.
        if temperature != 1.0 or sf_bias != 0.0 or epsilon != 0.0:
            logits[:, -1] -= sf_bias
            probs_county = torch.softmax(logits_county / temperature, dim=-1)
            probs_district = torch.softmax(logits_district / temperature, dim=-1)

            uniform_dist_probs_county = masks_county.float() / masks_county.sum(dim=-1, keepdim=True)
            uniform_dist_probs_district = masks_district.float() / masks_district.sum(dim=-1, keepdim=True)

            probs_county = (1 - epsilon) * probs_county + epsilon * uniform_dist_probs_county
            probs_district = (1 - epsilon) * probs_district + epsilon * uniform_dist_probs_district
            dist_county = UnsqueezedCategorical(probs=probs_county)
            dist_district = UnsqueezedCategorical(probs=probs_district)

            return dist_county, dist_district

        # LogEdgeFlows are greedy, as are most P_B.
        else:
            dist_county = UnsqueezedCategorical(logits=logits_county)
            dist_district = UnsqueezedCategorical(logits=logits_district)
            return dist_county, dist_district
        
class DistrictSampler(Sampler):
    def __init__(self, estimator):
        super().__init__(estimator)

    def sample_actions(self, env, states):
        est_output = self.estimator(states)

        dist_county, dist_district = self.estimator.to_probability_distribution(
            states, est_output
        )

        with torch.no_grad():
            county_action = dist_county.sample()
            district_action = dist_district.sample()
            action = torch.cat((county_action, district_action), dim=-1)
            action = env.actions_from_tensor(action)

        log_probs_county = dist_county.log_prob(county_action)
        log_probs_district = dist_district.log_prob(district_action)

        log_probs = log_probs_county + log_probs_district
        
        return action, log_probs, est_output
    
    def sample_trajectories(
        self,
        env: Env,
        n: Optional[int] = None,
        states: Optional[States] = None,
        conditioning: Optional[torch.Tensor] = None,
        save_estimator_outputs: bool = False,
        save_logprobs: bool = True,
        **policy_kwargs: Any,
    ) -> Trajectories:
        """Sample trajectories sequentially.

        Args:
            env: The environment to sample trajectories from.
            n: If given, a batch of n_trajectories will be sampled all
                starting from the environment's s_0.
            states: If given, trajectories would start from such states. Otherwise,
                trajectories are sampled from $s_o$ and n_trajectories must be provided.
            conditioning: An optional tensor of conditioning information.
            save_estimator_outputs: If True, the estimator outputs will be returned. This
                is useful for off-policy training with tempered policy.
            save_logprobs: If True, calculates and saves the log probabilities of sampled
                actions. This is useful for on-policy training.
            policy_kwargs: keyword arguments to be passed to the
                `to_probability_distribution` method of the estimator. For example, for
                DiscretePolicyEstimators, the kwargs can contain the `temperature`
                parameter, `epsilon`, and `sf_bias`. In the continuous case these
                kwargs will be user defined. This can be used to, for example, sample
                off-policy.

        Returns: A Trajectories object representing the batch of sampled trajectories.

        Raises:
            AssertionError: When both states and n_trajectories are specified.
            AssertionError: When states are not linear.
        """

        if states is None:
            assert n is not None, "Either kwarg `states` or `n` must be specified"
            states = env.reset(batch_shape=(n,))
            n_trajectories = n
        else:
            assert (
                len(states.batch_shape) == 1
            ), "States should have len(states.batch_shape) == 1, w/ no trajectory dim!"
            n_trajectories = states.batch_shape[0]

        if conditioning is not None:
            assert states.batch_shape == conditioning.shape[: len(states.batch_shape)]

        device = states.tensor.device

        dones = (
            states.is_initial_state
            if self.estimator.is_backward
            else states.is_sink_state
        )

        trajectories_states: List[States] = [deepcopy(states)]
        trajectories_actions: List[torch.Tensor] = []
        trajectories_logprobs: List[torch.Tensor] = []
        trajectories_is_valid: List[torch.Tensor] = []
        trajectories_dones = torch.zeros(
            n_trajectories, dtype=torch.long, device=device
        )
        trajectories_log_rewards = torch.zeros(
            n_trajectories, dtype=torch.float, device=device
        )

        step = 0
        all_estimator_outputs = []

        while not all(dones):
            actions = env.Actions.make_dummy_actions(batch_shape=(n_trajectories,))
            log_probs = torch.full(
                (n_trajectories,), fill_value=0, dtype=torch.float, device=device
            )
            # This optionally allows you to retrieve the estimator_outputs collected
            # during sampling. This is useful if, for example, you want to evaluate off
            # policy actions later without repeating calculations to obtain the env
            # distribution parameters.
            if conditioning is not None:
                masked_conditioning = conditioning[~dones]
            else:
                masked_conditioning = None

            valid_actions, actions_log_probs, estimator_outputs = self.sample_actions(
                env,
                states[~dones],
                **policy_kwargs,
            )
            if estimator_outputs is not None:
                # Place estimator outputs into a stackable tensor. Note that this
                # will be replaced with torch.nested.nested_tensor in the future.
                estimator_outputs_padded = torch.full(
                    (n_trajectories,) + estimator_outputs.shape[1:],
                    fill_value=-float("inf"),
                    dtype=torch.float,
                    device=device,
                )
                estimator_outputs_padded[~dones] = estimator_outputs
                all_estimator_outputs.append(estimator_outputs_padded)
            
            actions[~dones] = valid_actions

            is_valid = env.get_valid_actions(states, actions, self.estimator.is_backward)

            env.is_valid = is_valid

            if save_logprobs:
                # When off_policy, actions_log_probs are None.
                log_probs[~dones] = actions_log_probs
            trajectories_actions.append(actions)
            trajectories_logprobs.append(log_probs)
            trajectories_is_valid.append(is_valid)
            if self.estimator.is_backward:
                print('bw')
                new_states = env._backward_step(states, actions)
                sink_states_mask = torch.zeros_like(actions[:, 0], device=device)
                sink_states_mask = sink_states_mask & ~dones
            else:
                new_states = env._step(states, actions)
                sink_states_mask = actions.is_exit
                sink_states_mask = sink_states_mask | ~is_valid

            # Increment the step, determine which trajectories are finisihed, and eval
            # rewards.
            step += 1

            # fill sink states with -inf
            new_states.is_sink = sink_states_mask

            # new_dones means those trajectories that just finished. Because we
            # pad the sink state to every short trajectory, we need to make sure
            # to filter out the already done ones.
            new_dones = (
                new_states.is_initial_state
                if self.estimator.is_backward
                else sink_states_mask
            ) & ~dones
            trajectories_dones[new_dones & ~dones] = step
            try:
                trajectories_log_rewards[new_dones & ~dones] = env.log_reward(
                    states[new_dones & ~dones], is_valid
                )
            except NotImplementedError:
                trajectories_log_rewards[new_dones & ~dones] = torch.log(
                    env.reward(states[new_dones & ~dones])
                )
            states = new_states
            dones = dones | new_dones

            trajectories_states.append(deepcopy(states))

        trajectories_states = stack_states_district(trajectories_states)
        trajectories_actions = env.Actions.stack(trajectories_actions)
        trajectories_logprobs = (
            torch.stack(trajectories_logprobs, dim=0) if save_logprobs else None
        )
        trajectories_is_valid = torch.stack(trajectories_is_valid, dim=0)

        # TODO: use torch.nested.nested_tensor(dtype, device, requires_grad).
        if save_estimator_outputs:
            all_estimator_outputs = torch.stack(all_estimator_outputs, dim=0)

        trajectories = Trajectories(
            env=env,
            states=trajectories_states,
            conditioning=conditioning,
            actions=trajectories_actions,
            when_is_done=trajectories_dones,
            is_backward=self.estimator.is_backward,
            log_rewards=trajectories_log_rewards,
            log_probs=trajectories_logprobs,
            estimator_outputs=all_estimator_outputs if save_estimator_outputs else None,
            is_valid=trajectories_is_valid,
        )

        return trajectories
