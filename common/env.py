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


class DistrictEnv(Env):
    def __init__(
        self,
        json_file: str,
        device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        """Initialize our districting environment.
        """
        self.json_file = json_file
        self.district_graph = Graph('cpu')
        self.district_graph.graph_from_json(self.json_file)

        s0 = self.district_graph.get_full_state()

        dummy_action = torch.tensor([0,0], device=torch.device(device_str))
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

    def step(self, states: States, actions: Actions) -> torch.Tensor:
        """Take a step in the environment.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns the new states after taking the actions as a tensor of shape (*batch_shape, 2).
        """
        for _batch in range(states.tensor.shape[0]):
            if self.is_valid[_batch]:
                vertex_id, new_district_id = actions.tensor[_batch]
                self.graphs[_batch].change_vertex_district(prescinct_id=vertex_id.item(), new_district_id=new_district_id.item())
                states.tensor[_batch] = self.graphs[_batch].get_full_state()
            else:
                if actions.tensor[_batch][0] == self.district_graph.num_county:
                    continue
                self.graphs[_batch].taken_actions[actions.tensor[_batch][0], actions.tensor[_batch][1]] = 1

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
            if vertex_id == self.district_graph.num_county:
                valid_actions[_batch] = False
                continue
            old_district_id = self.graphs[_batch].state[vertex_id, 0]

            if self.graphs[_batch].vertex_on_border.get((vertex_id, old_district_id)) is None:
                valid_actions[_batch] = False
            
            graph_state, action_state = self.graphs[_batch].full_state_to_graph_state_taken_actions(states.tensor[_batch])
            if action_state[vertex_id, new_district_id] == 1:
                valid_actions[_batch] = False

        return valid_actions
    
    def log_reward(self, final_states: States, is_valid) -> torch.Tensor:
        """Log reward log of the environment.

        Args:
            final_states: The final states of the environment.

        Returns the log reward as a tensor of shape `batch_shape`.
        """
        #TO IMPLEMENT

        log_rewards = torch.zeros(final_states.batch_shape, device=final_states.device)
        log_rewards[is_valid] = 1.0
        log_rewards[~is_valid] = -100
        return log_rewards


    def reset(
            self,
            batch_shape: Optional[Union[int, Tuple[int]]] = None,
            random: bool = False,
            sink: bool = False,
            seed: int = None,
        ) -> States:
        """
        Instantiates a batch of initial states. random and sink cannot be both True.
        When random is true and seed is not None, environment randomization is fixed by
        the submitted seed for reproducibility.
        """
        assert not (random and sink)

        if random and seed is not None:
            torch.manual_seed(seed)

        if batch_shape is None:
            batch_shape = (1,)
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)

        
        states = self.States.from_batch_shape(
            batch_shape=batch_shape, random=random, sink=sink
        )
        
        self.graphs = []
        for i in range(states.tensor.shape[0]):
            #create all graph instances
            _graph = Graph('cpu')
            _graph.graph_from_json(self.json_file)
            states.tensor[i] = _graph.get_full_state()
            self.graphs.append(self.district_graph)

        return states
    

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
    
