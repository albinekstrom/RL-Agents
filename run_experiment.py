import argparse
import gymnasium as gym
import importlib.util
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
parser.add_argument("--init", type=str, help="Initialization method [realistic/optimistic]", default="realistic")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)

episodic = False
try:
    env = gym.make(args.env, is_slippery=True)
    print("Loaded ", args.env)
    episodic = True
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)
    episodic = False

action_dim = env.action_space.n
state_dim  = env.observation_space.n

N_RUNS     = 5          # Number of runs
N_EPISODES = 10_000     # Number of episodes per run
MAX_STEPS  = 10_000     # Maximum number of steps per episode (most useful in non-episodic tasks)

# Episodic environments
rewards = []
for run in range(N_RUNS):
    print(f"Run {run+1}/{N_RUNS}")
    agent = agentfile.Agent(state_dim, action_dim, args.init)

    run_rewards = []

    # Non-episodic
    if not episodic:
        observation = env.reset() # NOTE: Here observation is a tuple -- important for Sarsa to tell first step of episode

        for step in range(MAX_STEPS):
            action = agent.act(observation)
            observation, reward, done, truncated, info = env.step(action)
            agent.observe(observation, reward, done)

            run_rewards.append(reward)

    else: # Episodic
        for e in range(N_EPISODES):
            observation = env.reset() # NOTE: Here observation is a tuple -- important for Sarsa to tell first step of episode

            episode_rewards = []
            while True:
                action = agent.act(observation)
                observation, reward, done, truncated, info = env.step(action)
                agent.observe(observation, reward, done)

                episode_rewards.append(reward)

                if done or truncated:
                    break

            run_rewards.append(np.mean(episode_rewards))

    rewards.append(run_rewards)

# 2. Plot rewards
import pandas as pd
import matplotlib
from scipy import stats
import matplotlib.pyplot as plt

#matplotlib.use('TkAgg')    # Different backend sometimes necessary
window_size = 1000
all_ma_rewards = []
for run_rewards in rewards:
    ma_rewards = pd.Series(run_rewards) \
                    .rolling(window_size, min_periods=1) \
                    .mean() \
                    .tolist()
    all_ma_rewards.append(ma_rewards)
    plt.plot(ma_rewards, label="Moving average reward", c="Orange")
    if episodic:
        plt.xlabel("Episode")
    else:
        plt.xlabel("Step")
    plt.ylabel("Reward")

mean_ma_rewards = [np.mean(rewards) for rewards in zip(*all_ma_rewards)]
plt.plot(mean_ma_rewards, label="Average MA reward", c="Red")

plt.legend()
plt.show()

# Calculate standard error of the mean
std_err = stats.sem(all_ma_rewards)

# Calculate the 95% confidence interval
confidence_interval = std_err * stats.t.ppf((1 + 0.95) / 2.0, len(all_ma_rewards) - 1)

plt.figure(figsize=(10,5))
plt.errorbar(range(len(mean_ma_rewards)), mean_ma_rewards, yerr=confidence_interval, fmt='-', color='black',
             ecolor='lightgrey', elinewidth=2, capsize=0, label="Moving Avarage Rewards")
if episodic:
    plt.xlabel("Episode")
else:
    plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Moving Average Rewards with 95% confidence interval")

plt.legend()
plt.show()

# 3. Plot Q
agent.show_values()

# Plot greedy policy
agent.show_policy(args.env)

env.close()
