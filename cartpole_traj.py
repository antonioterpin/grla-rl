from cartpole_env import CartPole
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import multiprocessing as mp

def run_experiment(lambda_reg, seed, train_timesteps, num_train_envs, noise_values, device):
    # Set seed for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create the training environment and train the PPO agent.
    train_env = VecMonitor(CartPole(num_envs=num_train_envs, device=device, lambda_reg=lambda_reg))
    model = PPO("MlpPolicy", train_env, verbose=1, device='cuda', n_steps=8, n_epochs=20, learning_rate=1e-3, batch_size=256)
    model.learn(total_timesteps=train_timesteps, log_interval=10)
    model.save('{}_200k.zip'.format(lambda_reg))
    #model = PPO.load('{}.zip'.format(lambda_reg), env=train_env, verbose=1, device='cuda', n_steps=8, n_epochs=20, learning_rate=1e-3,
    #            batch_size=256)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Dictionary to store evaluation results: noise -> trajectory (list of obs[:,2] values)
    experiment_results = {}
    for noise in noise_values:
        # Create evaluation environment with a single environment instance.
        eval_env = VecMonitor(CartPole(
            num_envs=1,
            device=device,
            bias=noise,
            lambda_reg=lambda_reg  # pass lambda_reg if needed by your environment
        ))
        obs = eval_env.reset()
        # Record the trajectory of the third element of the observation.
        trajectory = [obs[0, 2]]
        done = False
        # Run one episode until done.
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
            trajectory.append(obs[0, 2] if not done else infos[0]['terminal_observation'][2])

        experiment_results[noise] = trajectory
        print(f"lambda_reg={lambda_reg}, noise variance={noise}, trajectory length={len(trajectory)}")

    return lambda_reg, experiment_results

def main():
    # --- Configuration ---
    lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    noise_values = np.linspace(-0.999, 0.999, 21)  # 21 noise values
    seed = 0  # single seed for all experiments
    train_timesteps = 200000
    num_train_envs = 256  # training environments
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare parameters for each lambda experiment.
    params = [(lmb, seed, train_timesteps, num_train_envs, noise_values, device)
              for lmb in lambda_values]

    # Run experiments concurrently using multiprocessing.
    with mp.Pool(processes=16) as pool:
        results_list = pool.starmap(run_experiment, params)

    # Aggregate results: aggregated_results[lambda_reg][noise] = trajectory.
    aggregated_results = {lmb: results for lmb, results in results_list}

    # --- Plotting ---
    # Create a subplot for each noise value.
    n_noise = len(noise_values)
    # Arrange subplots in a grid: here 3 rows x 7 columns works for 21 plots.
    ncols = 3
    nrows = 8
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows), sharex=False, sharey=False)
    axs = axs.flatten()

    # For each noise value, plot trajectories for different lambda values.
    for i, noise in enumerate(noise_values):
        ax = axs[i]
        for lambda_reg in lambda_values:
            trajectory = aggregated_results[lambda_reg][noise]
            ax.plot(range(len(trajectory)), trajectory, label=f"λ={lambda_reg}")
        ax.set_title("assumed/true p = {}%".format(int(100+noise*100)))
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Theta")
        ax.legend(fontsize='x-small', loc='upper right')

    # Hide any unused subplots.
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
