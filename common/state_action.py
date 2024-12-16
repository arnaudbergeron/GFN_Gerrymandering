import torch
from gfn.actions import Actions
from gfn.states import States, stack_states


class DistrictState(States):
    def __init__(self, tensor, is_sink):
        super().__init__(tensor)
        self.is_sink = is_sink

    @classmethod
    def from_batch_shape(
        cls, batch_shape: tuple[int], random: bool = False, sink: bool = False
    ) -> States:
        """Create a States object with the given batch shape.

        By default, all states are initialized to $s_0$, the initial state. Optionally,
        one can initialize random state, which requires that the environment implements
        the `make_random_states_tensor` class method. Sink can be used to initialize
        states at $s_f$, the sink state. Both random and sink cannot be True at the
        same time.

        Args:
            batch_shape: Shape of the batch dimensions.
            random (optional): Initalize states randomly.
            sink (optional): States initialized with s_f (the sink state).

        Raises:
            ValueError: If both Random and Sink are True.
        """
        if random and sink:
            raise ValueError("Only one of `random` and `sink` should be True.")

        if random:
            tensor = cls.make_random_states_tensor(batch_shape)
            cls.is_sink = torch.zeros(batch_shape, dtype=torch.bool)
        elif sink:
            tensor = cls.make_sink_states_tensor(batch_shape)
            cls.is_sink = torch.ones(batch_shape, dtype=torch.bool)
        else:
            tensor = cls.make_initial_states_tensor(batch_shape)
            cls.is_sink = torch.zeros(batch_shape, dtype=torch.bool)
        return cls(tensor, cls.is_sink)
    
    def __getitem__(
        self, index
    ) -> States:
        """Access particular states of the batch."""
        out = self.__class__(
            self.tensor[index], self.is_sink[index]
        )  # TODO: Inefficient - this might make a copy of the tensor!
        if self._log_rewards is not None:
            out.log_rewards = self._log_rewards[index]
        return out
    
    def extend(self, other):
        super().extend(other)
        self.is_sink = torch.cat([self.is_sink, other.is_sink])
    
    @property
    def is_sink_state(self) -> torch.Tensor:
        """Returns a tensor of shape `batch_shape` that is True for states that are $s_f$ of the DAG."""
        return self.is_sink
    

class DistrictActions(Actions):
    """Base class for actions for all GFlowNet environments.

    Each environment needs to subclass this class. A generic subclass for discrete
    actions with integer indices is provided Note that all actions need to have the
    same shape.

    Attributes:
        tensor: a batch of actions with shape (*batch_shape, *actions_ndims).
        batch_shape: the batch_shape from the input tensor.
    """
    @property
    def is_exit(self) -> torch.Tensor:
        """Returns a boolean tensor of shape `batch_shape` indicating whether the actions are exit actions."""
        exit_actions = self.tensor[...,0] == self.exit_action
        return exit_actions
    


def stack_states_district(states) -> States:
    """Stack a list of states into a single States object.

    Args:
        states: A list of States objects.

    Returns a single States object with the stacked states.
    """
    out = stack_states(states)
    is_sink_out = torch.stack([state.is_sink for state in states], dim=0)
    out.is_sink = is_sink_out
    return out
    
