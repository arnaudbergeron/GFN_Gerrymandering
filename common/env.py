import random
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
from gfn.states import States, stack_states
from gfn.utils.distributions import UnsqueezedCategorical
from torch.distributions import Categorical, Distribution

from common.graph import Graph
from common.state_action import DistrictActions, DistrictState
from utils.data_utils import (
    compute_compactness,
    compute_efficiency_gap,
    compute_partisan_bias,
    compute_population_entropy,
)


class DistrictEnv(Env):
    def __init__(
        self,
        json_file: str,
        device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        """Initialize our districting environment.
        """
        self.json_file = json_file
        self.district_graph = Graph(device_str)
        self.df_graphs = None
        self.graphs = None
        df = self.district_graph.graph_from_json(self.json_file)
        self.district_graph.get_border_vertices()
        s0 = self.district_graph.get_full_state()

        dummy_action = torch.tensor([1000,1000], device=torch.device(device_str))
        exit_action = self.district_graph.num_county

        n_actions = self.district_graph.num_county+1+self.district_graph.max_district_id
        self.n_actions = (n_actions, )
        self.action_shape = (2,)  # [vertex_id, old_district_id, new_district_id]

        super().__init__(
            s0=s0,
            state_shape=s0.shape,
            action_shape=self.action_shape,  # [vertex_id, old_district_id, new_district_id]
            dummy_action=dummy_action,
            exit_action=exit_action,
            preprocessor=DistrictPreProcessor(s0.shape, self.district_graph.num_county),
        )  # sf is -inf by default.

    def step(self, states: States, actions: Actions, idx_list) -> torch.Tensor:
        """Take a step in the environment.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns the new states after taking the actions as a tensor of shape (*batch_shape, 2).
        """
        for _idx, _batch in enumerate(idx_list):
            if self.is_valid[_batch]:
                vertex_id, new_district_id = actions.tensor[_idx]
                self.graphs[_batch].change_vertex_district(prescinct_id=vertex_id.item(), new_district_id=new_district_id.item()+1)
                states.tensor[_idx] = self.graphs[_batch].get_full_state()
                self.graphs[_batch].taken_actions[actions.tensor[_idx][0], actions.tensor[_idx][1]] = 1
            else:
                if actions.tensor[_idx][0] == self.district_graph.num_county:
                    continue
                self.graphs[_batch].taken_actions[actions.tensor[_idx][0], actions.tensor[_idx][1]] = 1

        return states.tensor

    def backward_step(self, states: States, actions: Actions) -> torch.Tensor:
        """Take a step in the environment in the backward direction.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns the new states after taking the actions as a tensor of shape (*batch_shape, 2).
        """
        for _batch in range(states.tensor.shape[0]):
            vertex_id, new_district_id = actions.tensor[_batch]
            previous_id = self.graphs[_batch].previous_state[vertex_id]
            self.graphs[_batch].change_vertex_district(prescinct_id=vertex_id, new_district_id=previous_id, backward=True)
            states.tensor[_batch] = self.graphs[_batch].state

        return states.tensor
    
    def is_action_valid(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        return True
    
    def is_exit_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Determines if the actions are exit actions.

        Args:
            actions: tensor of actions of shape (*batch_shape, *action_shape)

        Returns tensor of booleans of shape (*batch_shape)
        """
        return actions[:,0] == self.district_graph.num_county+1


    def get_valid_actions(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        # Can't take a backward step at the beginning of a trajectory.
        valid_actions = torch.ones(states.tensor.shape[0], dtype=torch.bool)
        # if torch.any(states[~actions.is_exit].is_initial_state) and backward:
        #     valid_actions[states[~actions.is_exit].is_initial_state] = False
        
        for _batch in range(states.tensor.shape[0]):
            vertex_id, new_district_id = actions.tensor[_batch]
            vertex_id = vertex_id.item()
            new_district_id = new_district_id.item()
            if vertex_id == self.district_graph.num_county:
                valid_actions[_batch] = False
                continue

            if self.graphs[_batch].vertex_on_border.get((vertex_id, new_district_id+1)) is None:
                valid_actions[_batch] = False
                continue
            
            graph_state, action_state = self.graphs[_batch].full_state_to_graph_state_taken_actions(states.tensor[_batch])
            if action_state[vertex_id, new_district_id] == 1:
                valid_actions[_batch] = False

        return valid_actions
    
    def log_reward(self, final_states: States, actions, is_valid, idx_original, step) -> torch.Tensor:
        """Log reward log of the environment.

        Args:
            final_states: The final states of the environment.

        Returns the log reward as a tensor of shape `batch_shape`.
        """
        #TO IMPLEMENT
        new_district_col = 'new_district_id'
        log_rewards = torch.zeros(final_states.batch_shape, device=final_states.device)
        for _idx, _batch in enumerate(idx_original):
            if not is_valid[_idx]:
                if actions.tensor[_idx][0] != self.district_graph.num_county:
                    log_rewards[_idx] = -5  #check if valid.
                else:
                    log_rewards[_idx] = 1

            # log_rewards[_idx] = 100*step
            state = self.graphs[_batch].state
            s0_reshaped, _ = self.graphs[_batch].full_state_to_graph_state_taken_actions(self.graphs[_batch].s0)
            s0_diff = (state[:,0].abs() != s0_reshaped[:,0].abs()).sum().item()
            s0_diff_reward = min(s0_diff, 50)
            

            _df = self.df_graphs[_batch]
            _df[new_district_col] = state[:,0].cpu().abs().numpy()

            partisan_biais = compute_partisan_bias(_df,  _df[new_district_col], dem_vote_col="pre_20_dem_bid", rep_vote_col="pre_20_rep_tru", v=0.5)
            compactness_mean, compactness_std = compute_compactness(_df, _df[new_district_col])
            compactness = compactness_mean - compactness_std

            eff_gap = compute_efficiency_gap(_df,  _df[new_district_col], dem_vote_col="pre_20_dem_bid", rep_vote_col="pre_20_rep_tru")
            pop_entropy =  compute_population_entropy(_df,  _df[new_district_col], population_col="pop")
            # log_rewards[_idx] += -10*partisan_biais + 10*compactness - 10*eff_gap + 10*pop_entropy + 25*s0_diff_reward
            # print('partisan_biais', partisan_biais, 'eff_gap', eff_gap, 'pop_entropy', pop_entropy)
            # if abs(partisan_biais) < 0.01:
            #     print('partisan_biais', partisan_biais, 'eff_gap', eff_gap, 'pop_entropy', pop_entropy)
            log_rewards[_idx] += 10*(0.5-abs(partisan_biais)) + 10*(1-abs(eff_gap)) + 10*pop_entropy + 10*s0_diff_reward + 10*compactness
            # log_rewards[_idx] += 10*s0_diff_reward


        return log_rewards


    def reset(
            self,
            batch_shape: Optional[Union[int, Tuple[int]]] = None,
            random_reset: bool = False,
            sink: bool = False,
            seed: int = None,
        ) -> States:
        """
        Instantiates a batch of initial states. random and sink cannot be both True.
        When random is true and seed is not None, environment randomization is fixed by
        the submitted seed for reproducibility.
        """
        assert not (random_reset and sink)

        if random_reset and seed is not None:
            torch.manual_seed(seed)

        if batch_shape is None:
            batch_shape = (1,)
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)

        
        states = self.States.from_batch_shape(
            batch_shape=batch_shape, random=random_reset, sink=sink, is_start=True
        )
        
        _graphs_list = []
        self.df_graphs = []
        for i in range(states.tensor.shape[0]):
            #create all graph instances
            if self.graphs is None:
                _graph = deepcopy(self.district_graph)
                num_random_to_do = 50
            else:
                # _graph = deepcopy(self.graphs[i])
                _graph = deepcopy(self.district_graph)
                num_random_to_do = 5

            # num_random = torch.randint(0, num_random_to_do, (1,)).item()
            num_random = 0
            for _ in range(num_random):
                random_change = random.choice(list(_graph.vertex_on_border.keys()))
                _graph.change_vertex_district(prescinct_id=random_change[0], new_district_id=random_change[1])
            
            _graph.s0 = _graph.get_full_state()
            _df = _graph.df
            states.tensor[i] = _graph.s0
            _graphs_list.append(_graph)
            self.df_graphs.append(_df)
        
        self.graphs = _graphs_list

        return states
    
    def _step(
        self,
        states: States,
        actions: Actions,
    ) -> States:
        """Core step function. Calls the user-defined self.step() function.

        Function that takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating sink states in the new batch.
        """
        assert states.batch_shape == actions.batch_shape
        new_states = states.clone()  # TODO: Ensure this is efficient!
        valid_states_idx: torch.Tensor = ~states.is_sink_state
        assert valid_states_idx.shape == states.batch_shape
        assert valid_states_idx.dtype == torch.bool
        valid_actions = actions[valid_states_idx]
        valid_states = states[valid_states_idx]

        if not self.validate_actions(valid_states, valid_actions):
            raise Exception(
                "Some actions are not valid in the given states. See `is_action_valid`."
            )

        new_sink_states_idx = actions.is_exit
        # new_states.tensor[new_sink_states_idx] = self.sf
        new_sink_states_idx = ~valid_states_idx | new_sink_states_idx
        assert new_sink_states_idx.shape == states.batch_shape

        not_done_states = new_states[~new_sink_states_idx]
        not_done_actions = actions[~new_sink_states_idx]
        mask_idx = (~new_sink_states_idx).nonzero().reshape(-1).tolist()

        new_not_done_states_tensor = self.step(not_done_states, not_done_actions, mask_idx)
        if not isinstance(new_not_done_states_tensor, torch.Tensor):
            raise Exception(
                "User implemented env.step function *must* return a torch.Tensor!"
            )

        new_states.tensor[~new_sink_states_idx] = new_not_done_states_tensor

        return new_states
    

    def make_states_class(self) -> type[States]:
        """The default States class factory for all Environments.

        Returns a class that inherits from States and implements assumed methods.
        The make_states_class method should be overwritten to achieve more
        environment-specific States functionality.
        """
        env = self

        class DefaultEnvState(DistrictState):
            """Defines a States class for this environment."""

            state_shape = env.state_shape
            s0 = env.s0
            sf = env.sf
            make_random_states_tensor = env.make_random_states_tensor

        return DefaultEnvState

    def make_actions_class(self) -> type[Actions]:
        """The default Actions class factory for all Environments.

        Returns a class that inherits from Actions and implements assumed methods.
        The make_actions_class method should be overwritten to achieve more
        environment-specific Actions functionality.
        """
        env = self

        class DefaultEnvAction(DistrictActions):
            action_shape = env.action_shape
            dummy_action = env.dummy_action
            exit_action = env.exit_action

        return DefaultEnvAction


class DistrictPreProcessor(Preprocessor):
    """Simple preprocessor applicable to environments with uni-dimensional states.
    This is the default preprocessor used."""
    def __init__(
        self,
        dim,
        district_output_dim: int,
    ) -> None:
        """Initialize the preprocessor.
        """
        super().__init__(output_dim=dim[0])
        self.district_output_dim = district_output_dim

    def preprocess(self, states: States) -> torch.Tensor:
        """Identity preprocessor. Returns the states as they are."""
        return (
            states.tensor.float()
        )
    
