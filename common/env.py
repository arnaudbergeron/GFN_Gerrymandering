from typing import Literal, Optional, Tuple, Union

import torch
from gfn.actions import Actions
from gfn.env import Env
from gfn.preprocessors import Preprocessor
from gfn.states import States

from common.graph import Graph


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

        dummy_action = torch.tensor([0,0,0], device=torch.device(device_str))
        exit_action = torch.tensor([-1,-1,-1], device=torch.device(device_str))
        self.n_actions = (3,)  # [vertex_id, old_district_id, new_district_id]
        super().__init__(
            s0=s0,
            state_shape=s0.shape,
            action_shape=(3,),  # [vertex_id, old_district_id, new_district_id]
            dummy_action=dummy_action,
            exit_action=exit_action,
            preprocessor=DistrictPreProcessor(s0.shape),
        )  # sf is -inf by default.

    def step(self, states: States, actions: Actions) -> torch.Tensor:
        """Take a step in the environment.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns the new states after taking the actions as a tensor of shape (*batch_shape, 2).
        """
        for _batch in range(states.tensor.shape[0]):
            vertex_id, old_district_id, new_district_id = actions.tensor[_batch]
            self.graphs[_batch].change_vertex_district(prescinct_id=vertex_id, new_district_id=new_district_id)
            states.tensor[_batch] = self.graphs[_batch].get_full_state()

        return states.tensor

    def backward_step(self, states: States, actions: Actions) -> torch.Tensor:
        """Take a step in the environment in the backward direction.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns the new states after taking the actions as a tensor of shape (*batch_shape, 2).
        """
        for _batch in range(states.tensor.shape[0]):
            vertex_id, old_district_id, new_district_id = actions.tensor[_batch]
            self.graphs[_batch].change_vertex_district(prescinct_id=vertex_id, new_district_id=old_district_id, backward=True)
            states.tensor[_batch] = self.graphs[_batch].state

        return states.tensor
    
    def is_action_valid(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        # Can't take a backward step at the beginning of a trajectory.
        if torch.any(states[~actions.is_exit].is_initial_state) and backward:
            return False
        
        for _batch in range(states.tensor.shape[0]):
            vertex_id, old_district_id, new_district_id = actions.tensor[_batch]

            if self.graphs[_batch].vertex_on_border.get((vertex_id, old_district_id)) is None:
                return False
            
            graph_state, action_state = self.graphs[_batch].full_state_to_graph_state_taken_actions(states.tensor[_batch])
            if action_state[vertex_id, new_district_id] == 1:
                return False

        return True
    
    def log_reward(self, final_states: States) -> torch.Tensor:
        """Log reward log of the environment.

        Args:
            final_states: The final states of the environment.

        Returns the log reward as a tensor of shape `batch_shape`.
        """
        #TO IMPLEMENT
        log_rewards = torch.zeros(final_states.batch_shape, device=final_states.device)
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


class DistrictPreProcessor(Preprocessor):
    """Simple preprocessor applicable to environments with uni-dimensional states.
    This is the default preprocessor used."""
    def __init__(
        self,
        dim,
    ) -> None:
        """Initialize the preprocessor.
        """
        super().__init__(output_dim=dim[0])

    def preprocess(self, states: States) -> torch.Tensor:
        """Identity preprocessor. Returns the states as they are."""
        return (
            states.tensor.float()
        )