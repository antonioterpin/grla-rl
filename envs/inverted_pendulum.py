import jax
import jax.numpy as jp
from brax.envs.inverted_pendulum import InvertedPendulum as OldInvertedPendulum
from envs.base import DiffParams, DiffEnv, StateWithParams
from dataclasses import dataclass
from jax import tree_util

@tree_util.register_pytree_node_class
@dataclass
class Params(DiffParams):
    gravity: jax.Array
    # body_mass: jax.Array

    def tree_flatten(self):
        # return (self.gravity, self.body_mass), None
        return (self.gravity,), None
        # return (self.body_mass,), None

    def to_array(self):
        """Flatten the parameters into a single array."""
        # return jp.concatenate((self.gravity.flatten(), self.body_mass.flatten()))
        return self.gravity.flatten()
        # return self.body_mass.flatten()
    
    @classmethod
    def randomize(cls, rng: jax.random.PRNGKey) -> "Params":
        """Randomize the parameters."""
        rng, key = jax.random.split(rng)
        # Randomize gravity and body mass in uniform range
        return Params(
            gravity = jp.array([
                0.0,
                0.0,
                9.81 + jax.random.uniform(key, shape=(), minval=-1.0, maxval=1.0),
            ]),
            # body_mass = jp.array([
            #     0,
            #     10.471975 + 5 * jax.random.uniform(key, shape=(), minval=-1.0, maxval=1.0),
            #     5.0185914 + 3 * jax.random.uniform(key, shape=(), minval=-1.0, maxval=1.0)
            # ])
        )

class InvertedPendulum(OldInvertedPendulum, DiffEnv):
    def __init__(self, desensitization=0.1, params_bias=0, backend='generalized', **kwargs):
        super().__init__(backend, **kwargs)
        self.desensitization = desensitization
        self.params_bias = params_bias

    def reset(self, rng: jax.random.PRNGKey) -> StateWithParams:
        """Reset the environment state and return the initial observation."""
        state = super().reset(rng)
        params = Params.randomize(rng)

        return StateWithParams(
            pipeline_state= state.pipeline_state,
            obs=self.add_params_to_obs(state.obs, params),
            reward=state.reward,
            done=state.done,
            params=params,
            info=state.info,
        )

    def step(self, state: StateWithParams, action: jax.Array) -> StateWithParams:
        """Run one timestep of the environment's dynamics."""

        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        # Unpack the state
        pipeline_state0 = state.pipeline_state
        params = state.params

        # Define a function that computes the observation and returns auxiliary data.
        def compute_obs(p: DiffParams):
            pipeline_state = self.diff_pipeline_step(p, pipeline_state0, action)
            obs = self._get_obs(pipeline_state)
            return obs, (obs, pipeline_state)

        # Use jax.jacobian with has_aux=True to get both the Jacobian and the aux data.
        sensitivity_penalty = 0.0
        if self.desensitization > 1e-7:
            jacobian_fn = jax.jacobian(compute_obs, has_aux=True)
            jac, (obs, pipeline_state) = jacobian_fn(params)
            sensitivity_penalty = jp.sum(jp.square(jac.to_array())) * self.desensitization
        else:
            _, (obs, pipeline_state) = compute_obs(params)

        # Add params to obs
        obs = self.add_params_to_obs(obs, params)

        # Use the jacobian in the reward calculation.
        reward = 1.0 - sensitivity_penalty
        done = jp.where(jp.abs(obs[1]) > 0.2, 1.0, 0.0)

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs, reward=reward, done=done,
            params=params,
        )
    
    def add_params_to_obs(self, obs, params):
        params_observation = params.to_array() + self.params_bias
        return jp.concatenate((obs, params_observation), axis=-1)