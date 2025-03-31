import jax
import jax.numpy as jnp
from jax import jacobian, tree_util
from dataclasses import dataclass
from typing import Tuple
from brax.envs.base import State, Env

@tree_util.register_pytree_node_class
@dataclass
class BallPlateState:
    # Ball state: 2D position and velocity.
    x: jnp.ndarray         # ball x-position
    y: jnp.ndarray         # ball y-position
    x_dot: jnp.ndarray     # ball x-velocity
    y_dot: jnp.ndarray     # ball y-velocity
    # Plate state: tilt angles and angular velocities about x and y axes.
    theta_x: jnp.ndarray   # plate tilt angle about the x-axis (radians)
    theta_y: jnp.ndarray   # plate tilt angle about the y-axis (radians)
    theta_dot_x: jnp.ndarray  # plate angular velocity about the x-axis
    theta_dot_y: jnp.ndarray  # plate angular velocity about the y-axis
    # Physical parameters for the episode.
    plate_radius: jnp.ndarray     # plate radius (m)
    ball_mass: jnp.ndarray        # ball mass (kg)
    plate_mass: jnp.ndarray       # plate mass (kg)
    friction_ball: jnp.ndarray    # friction coefficient between ball and plate
    friction_plate: jnp.ndarray   # damping/friction in plate rotation

    def tree_flatten(self):
        children = (
            self.x, self.y, self.x_dot, self.y_dot,
            self.theta_x, self.theta_y, self.theta_dot_x, self.theta_dot_y,
            self.plate_radius, self.ball_mass, self.plate_mass,
            self.friction_ball, self.friction_plate,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

class BallPlate(Env):
    def __init__(
        self,
        desensitization: float = 0.0,
        plate_radius_range: Tuple[float, float] = (5, 5.000001),
        ball_mass_range: Tuple[float, float] = (0.1, 0.5),
        plate_mass_range: Tuple[float, float] = (0.5, 1),
        friction_ball_range: Tuple[float, float] = (0.03, 0.07),
        friction_plate_range: Tuple[float, float] = (0.08, 0.12),
        augment: bool = True,
        noise_std: float = 0.0,
        max_torque: float = 1,  # maximum torque (N·m) applied to plate
    ):
        self.desensitization = desensitization
        self.plate_radius_range = plate_radius_range
        self.ball_mass_range = ball_mass_range
        self.plate_mass_range = plate_mass_range
        self.friction_ball_range = friction_ball_range
        self.friction_plate_range = friction_plate_range
        self.augment = augment
        self.noise_std = noise_std

        # Dynamics constants:
        self.gravity = 9.81   # gravitational acceleration (m/s^2)
        self.tau = 0.02       # time step (s)
        self.max_torque = max_torque  # clip the applied torque to this maximum

    @property
    def action_size(self):
        # Action now corresponds to 2D torques applied to the plate.
        return 2

    @property
    def observation_size(self):
        # Base observations: ball state (4 values) and plate state (4 values).
        base = 4 + 4
        # Augmented extra parameters: 7 physical parameters.
        extra = 7 if self.augment else 0
        return base + extra

    @property
    def backend(self) -> str:
        return "positional"

    def reset(self, rng) -> State:
        """
        Reset the environment. The ball is initialized near the center and the plate is level.
        All physical parameters are sampled uniformly from their respective ranges.
        """
        rng, rng_state, rng_radius, rng_ball_mass, rng_plate_mass, rng_friction_ball, rng_friction_plate = jax.random.split(rng, 7)
        # Sample ball state: [x, y, x_dot, y_dot] near zero.
        x, y, x_dot, y_dot = jax.random.uniform(
            rng_state, shape=(4,), minval=-0.05, maxval=0.05)
        # Initialize plate state: angles and angular velocities
        theta_x, theta_y, theta_dot_x, theta_dot_y = jax.random.uniform(
            rng_state, shape=(4,), minval=-0.05, maxval=0.05)
        # Sample physical parameters from their respective ranges.
        plate_radius = jax.random.uniform(
            rng_radius, shape=(), minval=self.plate_radius_range[0], maxval=self.plate_radius_range[1]
        )
        ball_mass = jax.random.uniform(
            rng_ball_mass, shape=(), minval=self.ball_mass_range[0], maxval=self.ball_mass_range[1]
        )
        plate_mass = jax.random.uniform(
            rng_plate_mass, shape=(), minval=self.plate_mass_range[0], maxval=self.plate_mass_range[1]
        )
        friction_ball = jax.random.uniform(
            rng_friction_ball, shape=(), minval=self.friction_ball_range[0], maxval=self.friction_ball_range[1]
        )
        friction_plate = jax.random.uniform(
            rng_friction_plate, shape=(), minval=self.friction_plate_range[0], maxval=self.friction_plate_range[1]
        )
        state = BallPlateState(
            x=x,
            y=y,
            x_dot=x_dot,
            y_dot=y_dot,
            theta_x=theta_x,
            theta_y=theta_y,
            theta_dot_x=theta_dot_x,
            theta_dot_y=theta_dot_y,
            plate_radius=plate_radius,
            ball_mass=ball_mass,
            plate_mass=plate_mass,
            friction_ball=friction_ball,
            friction_plate=friction_plate,
        )
        metrics = {}
        return State(
            pipeline_state=state,
            obs=self._get_obs(state),
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            metrics=metrics,
        )

    def _get_obs(self, state: BallPlateState) -> jnp.ndarray:
        """
        Compute the observation from the state.
        Base observations include the ball and plate states.
        If augment is True, the normalized plate radius and additional physical parameters are appended.
        """
        # Base: ball (x, y, x_dot, y_dot) and plate (theta_x, theta_y, theta_dot_x, theta_dot_y)
        base_obs = jnp.array([
            state.x, state.y, state.x_dot, state.y_dot,
            state.theta_x, state.theta_y, state.theta_dot_x, state.theta_dot_y
        ])
        if self.augment:
            # Normalize the plate radius.
            norm_plate_radius = (state.plate_radius - self.plate_radius_range[0]) / (
                self.plate_radius_range[1] - self.plate_radius_range[0]
            )
            # Normalize the ball mass.
            norm_ball_mass = (state.ball_mass - self.ball_mass_range[0]) / (
                self.ball_mass_range[1] - self.ball_mass_range[0]
            )
            # Normalize the plate mass.
            norm_plate_mass = (state.plate_mass - self.plate_mass_range[0]) / (
                self.plate_mass_range[1] - self.plate_mass_range[0]
            )
            # Normalize the friction coefficient between ball and plate.
            norm_friction_ball = (state.friction_ball - self.friction_ball_range[0]) / (
                self.friction_ball_range[1] - self.friction_ball_range[0]
            )
            # Normalize the damping/friction in plate rotation.
            norm_friction_plate = (state.friction_plate - self.friction_plate_range[0]) / (
                self.friction_plate_range[1] - self.friction_plate_range[0]
            )
            extra = jnp.array([
                norm_plate_radius,
                norm_ball_mass,
                norm_plate_mass,
                norm_friction_ball,
                norm_friction_plate
            ])
            # Add noise to the augmented observation
            noise = jax.random.normal(jax.random.PRNGKey(0), shape=extra.shape) * self.noise_std
            extra += noise
            obs = jnp.concatenate([base_obs, extra], axis=-1)
        else:
            obs = base_obs
        return obs

    def step(self, brax_state: State, action: jax.Array) -> State:
        """
        Simulate one time step.
        The action is a 2D torque (N·m) vector applied to the plate.
        Ball dynamics include nonlinear gravity (using sin), friction, and a centrifugal term due to plate rotation.
        Plate dynamics are simulated via rotational dynamics.
        """
        # Clip the torque action to the maximum allowed.
        action = jnp.clip(action, -self.max_torque, self.max_torque)
        state = brax_state.pipeline_state

        # Unpack ball state.
        x, y, x_dot, y_dot = state.x, state.y, state.x_dot, state.y_dot
        # Unpack plate state.
        theta_x, theta_y = state.theta_x, state.theta_y
        theta_dot_x, theta_dot_y = state.theta_dot_x, state.theta_dot_y

        def dynamics_fn(params):
            # Unpack physical parameters.
            plate_radius, ball_mass, plate_mass, friction_ball, friction_plate = params

            # --- Plate dynamics ---
            # Compute moment of inertia of a disk (about a horizontal axis): I = 0.5 * m_plate * (plate_radius)^2.
            I_plate = 0.5 * plate_mass * (plate_radius ** 2)
            torque_x, torque_y = action[0], action[1]
            theta_ddot_x = (torque_x - friction_plate * theta_dot_x) / I_plate
            theta_ddot_y = (torque_y - friction_plate * theta_dot_y) / I_plate
            new_theta_dot_x = theta_dot_x + self.tau * theta_ddot_x
            new_theta_dot_y = theta_dot_y + self.tau * theta_ddot_y
            new_theta_x = theta_x + self.tau * new_theta_dot_x
            new_theta_y = theta_y + self.tau * new_theta_dot_y

            # --- Ball dynamics ---
            a_gravity_x = self.gravity * jnp.sin(theta_x)
            a_gravity_y = self.gravity * jnp.sin(theta_y)
            a_friction_x = - (friction_ball / ball_mass) * x_dot
            a_friction_y = - (friction_ball / ball_mass) * y_dot
            centr_factor = theta_dot_x * y - theta_dot_y * x
            a_cent_x = theta_dot_y * centr_factor
            a_cent_y = - theta_dot_x * centr_factor
            a_ball_x = a_gravity_x + a_friction_x + a_cent_x
            a_ball_y = a_gravity_y + a_friction_y + a_cent_y
            new_x_dot = x_dot + self.tau * a_ball_x
            new_y_dot = y_dot + self.tau * a_ball_y
            new_x = x + self.tau * x_dot
            new_y = y + self.tau * y_dot

            new_state = jnp.array([
                new_x, new_y, new_x_dot, new_y_dot,
                new_theta_x, new_theta_y, new_theta_dot_x, new_theta_dot_y
            ])
            return new_state, new_state

        penalty = 0.0
        params = (
            state.plate_radius, state.ball_mass, state.plate_mass,
            state.friction_ball, state.friction_plate
        )
        if self.desensitization > 1e-7:
            # Compute the Jacobian of the dynamics function.
            jacobian_fn = jacobian(dynamics_fn, has_aux=True)
            jac, new_state = jacobian_fn(params)
            # jax.debug.print("Jacobian: {}", jac)
            penalty = jnp.sum(jnp.concatenate(jac) ** 2) * self.desensitization
        else:
            # If no regularization, just compute the new state.
            new_state = dynamics_fn(params)[0]

        # Unpack the new state.
        new_x, new_y, new_x_dot, new_y_dot, new_theta_x, new_theta_y, new_theta_dot_x, new_theta_dot_y = new_state
        new_state = BallPlateState(
            x=new_x,
            y=new_y,
            x_dot=new_x_dot,
            y_dot=new_y_dot,
            theta_x=new_theta_x,
            theta_y=new_theta_y,
            theta_dot_x=new_theta_dot_x,
            theta_dot_y=new_theta_dot_y,
            plate_radius=state.plate_radius,
            ball_mass=state.ball_mass,
            plate_mass=state.plate_mass,
            friction_ball=state.friction_ball,
            friction_plate=state.friction_plate,
        )

        distance = jnp.sqrt(new_x**2 + new_y**2)
        reward = 1.0 - penalty
        done = jnp.where(distance > state.plate_radius ** 2, 1.0, 0.0)

        return brax_state.replace(
            pipeline_state=new_state,
            obs=self._get_obs(new_state),
            reward=(1 - done) * reward,
            done=done
        )

if __name__ == "__main__":
    # Test the extended BallPlate environment with randomized parameters.
    env = BallPlate(
        desensitization=0.0,
        noise_std=0.0,
        plate_radius_range=(1.0, 1.5),
        ball_mass_range=(0.08, 0.12),
        plate_mass_range=(0.8, 1.2),
        friction_ball_range=(0.03, 0.07),
        friction_plate_range=(0.08, 0.12),
    )
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    xs, ys = [], []
    thetas_x, thetas_y = [], []
    for _ in range(1000):
        act_rng, rng = jax.random.split(rng)
        action = jax.random.uniform(act_rng, shape=(2,), minval=-0.05, maxval=0.05)
        state = env.step(state, action)
        if state.done:
            break
        xs.append(state.pipeline_state.x)
        ys.append(state.pipeline_state.y)
        thetas_x.append(state.pipeline_state.theta_x)
        thetas_y.append(state.pipeline_state.theta_y)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(xs, ys, label='Ball Position')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.title('Nonlinear Ball and Plate Dynamics with Randomized Parameters')
    circle1 = plt.Circle((0, 0), env.plate_radius_range[0], color='r', fill=False, linestyle='--', label='Min Plate Radius')
    circle2 = plt.Circle((0, 0), env.plate_radius_range[1], color='r', fill=False, linestyle='--', label='Max Plate Radius')
    plt.gca().add_artist(circle1)
    plt.gca().add_artist(circle2)
    plt.gca().set_xlim(-env.plate_radius_range[1] - 0.1, env.plate_radius_range[1] + 0.1)
    plt.gca().set_ylim(-env.plate_radius_range[1] - 0.1, env.plate_radius_range[1] + 0.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('ball_plate_randomized_parameters.png')
