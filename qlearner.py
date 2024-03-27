import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Agent():
    """Q-learning"""

    def __init__(self, state_space, action_space, initialization=None):
        # NOTE: state_space and action_space are just numpy arrays ("spaces.Discrete.n")
        self.state_space  = state_space
        self.action_space = action_space

        self.alpha  = 0.10  # Step size
        self.eps    = 0.05  # 5% exploration rate
        self.gamma  = 0.95  # 95% discount factor

        # Q table: |S| x |A| matrix
        if initialization == "high":    # Optimistic
            self.Q = np.ones([state_space, action_space]) * 1
        elif initialization == "low":   # Realistic
            self.Q = np.ones([state_space, action_space]) * 0.1
        else:   # Random
            self.Q = np.random.rand(state_space, action_space)

        self.state  = 0     # S
        self.action = 0     # A

    def act(self, observation):

        # NOTE: Observation type differs depending on whether it's the output of env.reset() or env.step()...
        # Set start state
        if type(observation) == tuple:
            self.state, _ = observation
        # NOTE: Else is not needed since it will have been updated in `observe` from previous step

        if np.random.rand() < self.eps:
            # Explore (random action)
            self.action = np.random.randint(self.action_space)
        else:
            # Exploit (argmax of Q)
            # Sample greedily from Q table
            self.action = np.argmax(self.Q[self.state, :])

        return self.action

    def observe(self, observation, reward, done):
        # NOTE: Observation is new state

        # Update Q table
        future = (not done) * np.max(self.Q[observation, :]) 
        self.Q[self.state, self.action] += self.alpha * (reward + self.gamma * future - self.Q[self.state, self.action])

        # Move to next state
        if not done:
            self.state = observation

    def show_values(self):
        print(self.Q)
        plot = sns.heatmap(self.Q, annot=True, cmap='Greens')
        plot.set_title("Q-learning value function")
        plt.show()

    def show_policy(self, env):
        policy = np.zeros([self.state_space, 1])
        for s in range(self.state_space):
            policy[s] = np.argmax(self.Q[s, :])
        if env == "FrozenLake-v1":
            policy = policy.reshape([4, 4])
        plot = sns.heatmap(policy, annot=True, cmap="Reds")
        plot.set_title("Q-learning greedy policy")
        plt.show()
