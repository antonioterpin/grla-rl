from typing import Any
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices
from triton.language import dtype
import multiprocessing as mp


class CartPole(VecEnv):
    """
    A vectorized CartPole environment that uses PyTorch for simulation.
    This implementation follows the dynamics of the classic CartPole (discrete action) problem.
    """

    def __init__(
            self,
            num_envs: int,
            device: str = "cuda",
            max_steps: int = 500,
            lambda_reg=0.0,
            g_x=0.0,
            m_p_range=(0.075, 0.125),
            m_c_range=(0.75, 1.25),
            l_range=(0.375, 0.625),
            augment=True,
            integration='euler',
            theta_rewards=(),
            bias=0.0,
            seed=0,
    ):
        # Define observation and action spaces matching gym's CartPole.
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4 + augment * 3,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)

        #self.rng = np.random.default_rng(seed)
        #torch.manual_seed(seed)

        self.num_envs = num_envs
        self.device = device
        self.max_steps = max_steps
        self.lambda_reg = lambda_reg
        self.m_p_range = m_p_range
        self.m_c_range = m_c_range
        self.l_range = l_range
        self.augment = augment
        self.integration = integration

        # Step counts
        self._step_counts = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        # Define dynamics parameters as torch tensors with gradients enabled.
        self.gravity = torch.tensor(9.8, dtype=torch.float32, device=self.device, requires_grad=True)
        self.g_x = torch.full((num_envs,), g_x, dtype=torch.float32, device=self.device, requires_grad=True)  # horizontal gravity
        self.masscart = m_c_range[0] + (m_c_range[1] - m_c_range[0]) * torch.rand((num_envs,), dtype=torch.float32, device=self.device, requires_grad=True)
        self.masspole = m_p_range[0] + (m_p_range[1] - m_p_range[0]) * torch.rand((num_envs,), dtype=torch.float32, device=self.device, requires_grad=True)
        self.length = l_range[0] + (l_range[1] - l_range[0]) * torch.rand((num_envs,), dtype=torch.float32, device=self.device, requires_grad=True)
        # Note: total_mass and polemass_length are computed on-the-fly in step_wait.

        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Termination thresholds (as in gym's CartPole)
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * np.pi / 360  # ~0.20944 rad

        # Placeholder for asynchronous actions
        self._actions = None

        # For rendering with pygame.
        self.screen = None
        self.clock = None
        self.isopen = True
        self.screen_width = 600
        self.screen_height = 400

        self.theta_rewards = theta_rewards

        self.bias = bias

        self.s_norm = 0.25
        self.max_s = 0.0

        # Initialize the state for all environments.
        self.reset()

    def reset(self):
        self._step_counts = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        # In gym's CartPole, the state is initialized uniformly in [-0.05, 0.05]
        low, high = -0.05, 0.05
        self.state = (high - low) * torch.rand((self.num_envs, 4), device=self.device, requires_grad=True) + low
        self.masspole = self.m_p_range[0] + (self.m_p_range[1] - self.m_p_range[0]) * torch.rand((self.num_envs,),
                                                                                                 dtype=torch.float32,
                                                                                                 device=self.device,
                                                                                                 requires_grad=True)
        self.masscart = self.m_c_range[0] + (self.m_c_range[1] - self.m_c_range[0]) * torch.rand((self.num_envs,),
                                                                                                 dtype=torch.float32,
                                                                                                 device=self.device,
                                                                                                 requires_grad=True)
        self.length = self.l_range[0] + (self.l_range[1] - self.l_range[0]) * torch.rand((self.num_envs,),
                                                                                         dtype=torch.float32,
                                                                                         device=self.device,
                                                                                         requires_grad=True)

        return self._get_obs()

    def step_async(self, actions):
        """
        Store the actions for the next step.
        :param actions: A NumPy array of shape (num_envs,) or (num_envs, 1) with discrete actions {0, 1}.
        """
        # Convert actions to a torch tensor (flattened) on the correct device.
        self._actions = torch.tensor(actions, device=self.device).view(self.num_envs)

    def step_wait(self):
        """
        Compute one step of the environment using the stored actions.
        Returns observations, rewards, done flags, and info dictionaries.
        """
        # Convert action to force.
        force = self._actions * self.force_mag

        # Unpack state components: [x, x_dot, theta, theta_dot]
        x = self.state[:, 0]
        x_dot = self.state[:, 1]
        theta = self.state[:, 2]
        theta_dot = self.state[:, 3]

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        # Compute the differentiable parameters on the fly.
        total_mass = self.masscart + self.masspole
        polemass_length = self.masspole * self.length

        temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass

        # Modified angular acceleration including horizontal gravity:
        thetaacc = ((self.gravity * sintheta + self.g_x * costheta) - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / total_mass)
        )
        # x-acceleration includes the horizontal gravity as well.
        xacc = temp - polemass_length * thetaacc * costheta / total_mass + self.g_x

        # Euler integration.  TODO: test with semiimplicit
        if self.integration == 'euler':
            x = x + self.tau * x_dot
            theta = theta + self.tau * theta_dot
            x_dot = x_dot + self.tau * xacc
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            theta_dot = theta_dot + self.tau * thetaacc
            x = x + self.tau * x_dot
            theta = theta + self.tau * theta_dot

        new_state = torch.stack((x, x_dot, theta, theta_dot), dim=1)

        # Sensitivity computation.
        s_squared = np.zeros((self.num_envs,), dtype=np.float32)
        for i in ([1, 3] if self.integration == 'euler' else [0, 1, 2, 3]):
            for p in [self.length, self.masspole, self.masscart]:
                s = p * torch.autograd.grad(new_state[:, i], p, grad_outputs=torch.ones_like(x), retain_graph=True)[0]
                s_squared += s.detach().cpu().numpy() ** 2.0 / self.s_norm

        # TODO: TEMPORARY JUST TO TEST
        if s_squared.max() >= self.max_s:
            #print('maximum s: {}'.format(s_squared.max()))
            self.max_s = s_squared.max()

        # Determine which environments are done.
        done_mask = ((x < -self.x_threshold) | (x > self.x_threshold) |
                     (theta < -self.theta_threshold_radians) | (theta > self.theta_threshold_radians))

        # Update step counts and enforce time limit.
        self._step_counts += 1
        time_limit_done = self._step_counts >= self.max_steps

        # Combine done flags.
        done_all = done_mask | time_limit_done
        dones = done_all.detach().cpu().numpy()

        # Reward is 1 per step minus a regularization term.
        rewards = np.ones(self.num_envs, dtype=np.float32)
        for t in self.theta_rewards:
            rewards += (np.abs(theta.detach().cpu().numpy()) <= (t * np.pi / 180.0)).astype(np.float32)
        rewards -= self.lambda_reg * s_squared

        # Get observations.
        self.state = new_state.detach().clone().requires_grad_(True)
        obs = self._get_obs()

        infos = [{} for _ in range(self.num_envs)]
        if done_all.any():
            for i in range(self.num_envs):
                if done_all[i]:
                    infos[i]['terminal_observation'] = obs[i]
                    if time_limit_done[i]:
                        infos[i]['TimeLimit.truncated'] = True
            num_reset = int(done_all.sum().item())
            reset_states = (torch.rand((num_reset, 4), device=self.device) *
                            (0.05 - (-0.05)) + (-0.05))
            new_state[done_all] = reset_states
            self.masspole = self.masspole.detach().clone()
            self.masspole[done_all] = self.m_p_range[0] + (self.m_p_range[1] - self.m_p_range[0]) * torch.rand((num_reset,), device=self.device)
            self.masspole.requires_grad_(True)
            self.masscart = self.masscart.detach().clone()
            self.masscart[done_all] = self.m_c_range[0] + (self.m_c_range[1] - self.m_c_range[0]) * torch.rand((num_reset,), device=self.device)
            self.masscart.requires_grad_(True)
            self.length = self.length.detach().clone()
            self.length[done_all] = self.l_range[0] + (self.l_range[1] - self.l_range[0]) * torch.rand((num_reset,), device=self.device)
            self.length.requires_grad_(True)
            self._step_counts[done_all] = 0

        self.state = new_state.detach().clone().requires_grad_(True)
        return self._get_obs(), rewards, dones, infos

    def step(self, actions):
        """
        A synchronous step.
        """
        self.step_async(actions)
        return self.step_wait()

    def _get_obs(self, normalize=True):
        state, m_p, m_c, l = (self.state.detach().cpu().numpy(),
                              self.masspole.detach().cpu().numpy(),
                              self.masscart.detach().cpu().numpy(),
                              self.length.detach().cpu().numpy())
        m_p = (1 + self.bias) * m_p
        m_c = (1 + self.bias) * m_c
        l = (1 + self.bias) * l
        if normalize:
            m_p = (m_p - self.m_p_range[0]) / (self.m_p_range[1] - self.m_p_range[0] + 1e-10)
            m_c = (m_c - self.m_c_range[0]) / (self.m_c_range[1] - self.m_c_range[0] + 1e-10)
            l = (l - self.l_range[0]) / (self.l_range[1] - self.l_range[0] + 1e-10)
        return np.column_stack((state, m_p, m_c, l)) if self.augment else state

    def render(self, mode='human'):
        """
        Render the first environment instance using pygame.
        """
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("TorchCartPoleVecEnv")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.screen.fill((255, 255, 255))  # white background

        state = self.state[0].detach().cpu().numpy()  # [x, x_dot, theta, theta_dot]
        cartx = state[0]
        theta = state[2]

        scale = self.screen_width / (2 * self.x_threshold * 1.2)  # extra margin
        cart_x_pixel = int(self.screen_width / 2 + cartx * scale)
        cart_y_pixel = int(self.screen_height * 0.75)

        cart_width = 50
        cart_height = 30

        cart_rect = pygame.Rect(0, 0, cart_width, cart_height)
        cart_rect.center = (cart_x_pixel, cart_y_pixel)
        pygame.draw.rect(self.screen, (0, 0, 0), cart_rect)

        pole_length = 100  # in pixels
        pole_width = 10

        pole_x = cart_x_pixel + pole_length * np.sin(theta)
        pole_y = cart_y_pixel - cart_height // 2 - pole_length * np.cos(theta)

        pygame.draw.line(self.screen, (255, 0, 0), (cart_x_pixel, cart_y_pixel - cart_height // 2), (int(pole_x), int(pole_y)), pole_width)

        ground_y = int(self.screen_height * 0.75 + cart_height // 2)
        pygame.draw.line(self.screen, (0, 0, 0), (0, ground_y), (self.screen_width, ground_y), 2)

        pygame.display.flip()
        self.clock.tick(50)

        return None

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return [False] * self.num_envs if indices is None else [False for _ in indices]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        return [False] * self.num_envs if indices is None else [False for _ in indices]

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        pass
