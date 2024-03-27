import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Agent(object):
    """Expected SARSA"""

    def __init__(self, state_space, action_space, initialization=None):
        self.action_space   = action_space
        self.state_space    = state_space

        self.alpha  = 0.10  # Step size
        self.eps    = 0.05  # 5% exploration rate
        self.gamma  = 0.95  # 95% discount factor

        if initialization == "optimistic":
            self.Q = np.ones([state_space, action_space]) * 1
        elif:
            self.Q = np.ones([state_space, action_space]) * 0.1
        else:
            self.Q = np.random.rand(state_space, action_space)

        self.state  = 0    # S
        self.action = 0    # A

    def act(self, observation):
        
        if np.random.rand() < self.eps:
            self.action = np.random.randint(self.action_space)
        else:
            self.action = np.argmax(self.Q[self.state, :])
        
        return self.action
        
    def observe(self, observation, reward, done):

        # Compute expected future return
        future = 0
        if not done:
            for a in range(self.action_space):
                # Exploration (eps) + Exploitation (1-eps) -- Exploitation only if best action
                policy_prob = self.eps * 1/self.action_space + ((1-self.eps) if np.argmax(self.Q[observation, :]) == a else 0)
                future += policy_prob * self.Q[observation, a]

        self.Q[self.state, self.action] += self.alpha * (reward + self.gamma * future - self.Q[self.state, self.action])

        if not done:
            self.state = observation

    def show_values(self):
        print(self.Q)
        plot = sns.heatmap(self.Q, annot=True, cmap='Greens')
        plot.set_title("Expected SARSA value function")
        plt.show()

    def show_policy(self, env):
        policy = np.zeros([self.state_space, 1])
        for s in range(self.state_space):
            policy[s] = np.argmax(self.Q[s, :])
        if env == "FrozenLake-v1":
            policy = policy.reshape([4, 4])
        plot = sns.heatmap(policy, annot=True, cmap="Reds")
        plot.set_title("Expected SARSA greedy policy")
        plt.show()
