import argparse
import jax
import jax.numpy as jnp
import functools
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from envs.cartpole import CartPole
from envs.ballplate import BallPlate
from brax.training.agents.ppo import train as ppo
import yaml

# Parse arguments
parser = argparse.ArgumentParser(description="Train and test inverted pendulum model.")
parser.add_argument('--env', type=str, default='BallPlate', help='Environment to use (CartPole or BallPlate).')
parser.add_argument('--cuda_devices', type=str, default='0', help='CUDA visible devices.')
args = parser.parse_args()

# Set CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

# Read params from config
env_name = args.env
config = yaml.safe_load(open(f'configs/{env_name}.yaml'))

REGULARIZERS = config['experiment']['regularizers']
NOISE_STDS = config['experiment']['noise']

if env_name == 'CartPole':
    env_from_reg_and_bias = lambda reg, bias: CartPole(desensitization=reg, noise_std=bias)
elif env_name == 'BallPlate':
    env_from_reg_and_bias = lambda reg, bias: BallPlate(desensitization=reg, noise_std=bias)
else:
    raise ValueError(f"Unsupported environment: {env_name}")

# List to collect experiment results
results_data = []

for idx_reg, reg in enumerate(REGULARIZERS):
    print(f'Training with regularizer {reg}')
    # train with regularizer
    env = env_from_reg_and_bias(reg, 0)

    # Define a progress callback function.
    def progress_callback(n_steps, metrics):
        print((
            f"Num steps: {n_steps}"
            +
            f" | Reward: {metrics['eval/episode_reward']:.4f}"
        ))
    train_fn = functools.partial(ppo.train, **config['ppo'])

    t_start = time.time()
    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress_callback)
    print(f'Training took {time.time() - t_start:.2f} seconds')

    # Test with error in params estimation
    for idx_bias, bias in enumerate(NOISE_STDS):
        print(f'Testing model regularized by {reg} with bias {bias}')
        env = env_from_reg_and_bias(0, bias)

        # JIT compile the environment functions.
        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)
        # Build a deterministic inference function.
        # (Assume that make_inference_fn accepts a 'deterministic' flag.)
        jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
        jit_inference_fn = functools.partial(
            jit_inference_fn, key_sample=jax.random.PRNGKey(0))

        # Define a rollout function that runs one episode given a seed.
        def rollout(seed):
            # Each rollout gets its own PRNG key (unused here in a deterministic policy)
            rng = jax.random.PRNGKey(seed)
            state = jit_env_reset(rng=rng)
            cumulative_reward = 0.0

            # The inner loop is vectorized using jax.lax.scan.
            def step_fn(carry, _):
                state, cumulative_reward = carry
                # Compute action deterministically: e.g. take argmax of logits.
                act, _ = jit_inference_fn(state.obs)
                state = jit_env_step(state, act)
                cumulative_reward += state.reward
                return (state, cumulative_reward), None

            # Run for a fixed number of steps (DURATION) without early stopping.
            # (In a fully vectorized version, you can’t break out early;
            # alternatively, you can use a while_loop and mask updates after 'done'.)
            (final_state, cumulative_reward), _ = jax.lax.scan(
                step_fn, (state, cumulative_reward), None, length=config['ppo']['episode_length']
            )
            return cumulative_reward

        # Vectorize rollout over 1024 different seeds.
        seeds = jnp.arange(1024)
        results = jax.vmap(rollout)(seeds)

        mean_reward = jnp.mean(results)
        std_reward = jnp.std(results)

        print(f"Finished testing model with regularizer {reg} and bias {bias}.")
        print(f"Means: {mean_reward:.4f}")
        print(f"Stds: {std_reward:.4f}")

        # Collect results into our list
        results_data.append({
            "regularizer": reg,
            "bias": float(bias),
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        })

# Convert the results to a DataFrame and save to CSV.
df_results = pd.DataFrame(results_data)
csv_filename = f"results/{env_name}.csv"
df_results.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")

# Plotting the results
plt.figure(figsize=(8, 6))
# Plot mean rewards with error bars for each regularizer value.
for reg in sorted(df_results["regularizer"].unique()):
    sub_df = df_results[df_results["regularizer"] == reg]
    plt.errorbar(sub_df["bias"], sub_df["mean_reward"], yerr=sub_df["std_reward"],
                 marker='o', linestyle='-', capsize=5, label=f"Regularizer {reg}")

plt.xlabel("Noise Standard Deviation (Bias)")
plt.ylabel("Mean Reward")
plt.title("Mean Reward vs. Noise Bias for Different Regularizers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_filename = f"results/{env_name}.png"
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")
plt.show()