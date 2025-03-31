import jax
from brax import base
from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Any
from flax import struct
from brax.envs.base import State, Observation
from typing import Dict, Optional

@dataclass
class DiffParams:
    @abstractmethod
    def magnitude(self):
        """Compute the magnitude of the parameters."""
        pass

    @abstractmethod
    def randomize(self, rng):
        """Compute a randomized version of the parameters."""
        pass

    @abstractmethod
    def tree_flatten(self):
        pass

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    

@struct.dataclass
class StateWithParams():
    params: DiffParams
    pipeline_state: Optional[base.State]
    obs: Observation
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)
    

class DiffEnv:
    def diff_pipeline_step(
      self, params: DiffParams, pipeline_state: Any, action: jax.Array) -> base.State:
        """Takes a physics step using the physics pipeline."""

        def f(state, _):
            sys = self.sys.replace(
                **asdict(params),
            )
            return (
                self._pipeline.step(sys, state, action, self._debug),
                None,
            )

        return jax.lax.scan(f, pipeline_state, (), self._n_frames)[0]