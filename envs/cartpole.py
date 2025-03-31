import jax
import jax.numpy as jnp
from jax import jacobian
from dataclasses import dataclass
from typing import Tuple
from brax.envs.base import State, Env
from jax import tree_util

@tree_util.register_pytree_node_class
@dataclass
class CartPoleState:
    # State variables
    x: jnp.ndarray         # cart position
    x_dot: jnp.ndarray     # cart velocity
    theta: jnp.ndarray     # pole angle
    theta_dot: jnp.ndarray # pole angular velocity
    # Parameters (for each env instance)
    masspole: jnp.ndarray  # pole mass
    length: jnp.ndarray    # pole length
    rng_noise: jax.random.PRNGKey # RNG for noise

    def tree_flatten(self):
        return (self.x, self.x_dot, self.theta, self.theta_dot,
                self.masspole, self.length, self.rng_noise), None
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

class CartPole(Env):
    def __init__(
        self,
        desensitization: float = 0.0,
        m_p_range: Tuple[float, float] = (10, 30),
        l_range: Tuple[float, float] = (0.3, 0.7),
        augment: bool = True,
        noise_std: float = 0.0):
        self.desensitization = desensitization
        self.m_p_range = m_p_range
        self.l_range = l_range
        self.augment = augment
        self.noise_std = noise_std

        # Dynamics constants
        self.gravity = -9.81
        self.masscart = 1.0
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Termination thresholds (as in gym’s CartPole)
        self.x_threshold = 2.4
        self.theta_threshold = 12 * 2 * jnp.pi / 360  # ~0.20944 rad

    @property
    def action_size(self):
        return 1
    
    @property
    def observation_size(self):
        return 4

    @property
    def backend(self) -> str:
        return "positional"

    def reset(self, rng) -> State:
        """
        Reset the environment. Returns the initial state and observation.
        """
        rng, rng_state, rng_masspole, rng_length, rng_noise = jax.random.split(rng, 5)
        # Sample the four state variables uniformly in [-0.05, 0.05]
        state_sample = jax.random.uniform(
            rng_state, shape=(4,), minval=-0.05, maxval=0.05)
        x, x_dot, theta, theta_dot = state_sample
        masspole = jax.random.uniform(rng_masspole, shape=(),
                                      minval=self.m_p_range[0], maxval=self.m_p_range[1])
        length = jax.random.uniform(rng_length, shape=(),
                                    minval=self.l_range[0], maxval=self.l_range[1])
        state = CartPoleState(x, x_dot, theta, theta_dot, masspole, length, rng_noise)
        metrics = {}
        return State(
            pipeline_state=state,
            obs=self._get_obs(state),
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            metrics=metrics,
        )

    def _get_obs(self, state: CartPoleState) -> jnp.ndarray:
        """
        Compute the observation from the state.
        If augment is True, the normalized masspole and length are appended.
        """
        obs = jnp.stack([state.x, state.x_dot, state.theta, state.theta_dot], axis=-1)
        if self.augment:
            norm_masspole = (state.masspole - self.m_p_range[0]) / (self.m_p_range[1] - self.m_p_range[0])
            norm_length = (state.length - self.l_range[0]) / (self.l_range[1] - self.l_range[0])
            extra = jnp.stack([norm_masspole, norm_length], axis=-1)

            # Add noise to the augmented observation
            noise = jax.random.normal(state.rng_noise, shape=extra.shape) * self.noise_std
            extra += noise

            obs = jnp.concatenate([obs, extra], axis=-1)
        return obs

    def step(
        self,
        brax_state: State,
        action: jax.Array,
    ) -> State:
        """
        Take a simulation step given the current state and discrete action.
        """
        # Convert discrete action into force.
        force = jnp.clip(action, -1, 1) * self.force_mag
        
        # Unpack state
        state = brax_state.pipeline_state
        x, x_dot, theta, theta_dot = state.x, state.x_dot, state.theta, state.theta_dot
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        # Dynamics
        def dynamics_fn(params):
            L, mp = params[0], params[1]
            tot_mass = self.masscart + mp

            tmp = (force + L * mp * theta_dot**2 * sintheta) / tot_mass
            
            theta_ddot = (self.gravity * sintheta - costheta * tmp) / (
                L * (4.0/3.0 - mp * costheta**2 / tot_mass)
            )
            x_ddot = tmp - (mp * L * theta_ddot * costheta) / tot_mass
            
            theta_dot_new = theta_dot + self.tau * theta_ddot
            x_dot_new = x_dot + self.tau * x_ddot
            x_new = jnp.array([x + self.tau * x_dot])
            theta_new = jnp.array([theta + self.tau * theta_dot])

            new_state = jnp.concatenate([x_new, x_dot_new, theta_new, theta_dot_new])
            return new_state, new_state
        
        penalty = 0.0
        params = (state.length, state.masspole)
        if self.desensitization > 1e-7:
            # Compute the Jacobian of the dynamics function.
            jacobian_fn = jacobian(dynamics_fn, has_aux=True)
            jac, new_state = jacobian_fn(params)
            jac = jnp.concatenate(jac)
            d = jac.shape[0]
            penalty = jnp.sum(jac ** 2) / d * self.desensitization
        else:
            # If no regularization, just compute the new state.
            new_state = dynamics_fn(params)[0]


        # Construct new state
        x_new, x_dot_new, theta_new, theta_dot_new = new_state
        new_state = CartPoleState(
            x=x_new,
            x_dot=x_dot_new,
            theta=theta_new,
            theta_dot=theta_dot_new,
            masspole=state.masspole,
            length=state.length,
            rng_noise=state.rng_noise,
        )

        obs = self._get_obs(new_state)
        done = jnp.where(
            jnp.abs(new_state.x) > self.x_threshold,
            1.0,
            jnp.where(jnp.abs(new_state.theta) > self.theta_threshold, 1.0, 0.0),
        )

        return brax_state.replace(
            pipeline_state=new_state,
            obs=obs,
            reward=(1.0 - done) * jnp.float32(1.0 - penalty),
            done=done
        )
    
if __name__ == "__main__":
    # Test the CartPole environment
    env = CartPole(desensitization=0, noise_std=0)
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    xs = []
    thetas = []
    
    for _ in range(1000):
        act_rng, rng = jax.random.split(rng)
        action = jax.random.uniform(act_rng, shape=(1,), minval=0, maxval=0)
        state = env.step(state, action)
        if state.done:
            break
        xs.append(state.pipeline_state.x)
        thetas.append(state.pipeline_state.theta)

    import matplotlib.pyplot as plt
    plt.plot(xs, label='Cart Position')
    plt.plot(thetas, label='Pole Angle')
    # plot threshold lines
    plt.axhline(env.x_threshold, color='r', linestyle='--', label='X Threshold')
    plt.axhline(-env.x_threshold, color='r', linestyle='--')
    plt.axhline(env.theta_threshold, color='g', linestyle='--', label='Theta Threshold')
    plt.axhline(-env.theta_threshold, color='g', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.title('CartPole Dynamics')
    plt.savefig('cartpole_dynamics.png')