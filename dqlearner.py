import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Agent(object):
    """Double Q-learning"""

    def __init__(self, state_space, action_space, initialization=None):
        self.action_space = action_space
        self.state_space = state_space

        self.alpha  = 0.10  # Step size
        self.eps    = 0.05  # 5% exploration rate
        self.gamma  = 0.95  # 95% discount factor

        # Initialize two Q
        if initialization == "high":    # Optimistic
            self.Q1 = np.ones([state_space, action_space]) * 1
            self.Q2 = np.ones([state_space, action_space]) * 1
        elif initialization == "low":   # Realistic
            self.Q1 = np.ones([state_space, action_space]) * 0.1
            self.Q2 = np.ones([state_space, action_space]) * 0.1
        else:   # Random
            self.Q1 = np.random.rand(state_space, action_space)
            self.Q2 = np.random.rand(state_space, action_space)
        
        self.state  = 0     # S
        self.action = 0     # A

    def act(self, observation):

        if type(observation) == tuple:
            self.state, _ = observation
        
        if np.random.rand() < self.eps:
            # Explore
            self.action = np.random.randint(self.action_space)
        else:
            # Exploit
            self.action = np.argmax(self.Q1[self.state, :] + self.Q2[self.state, :])

        return self.action

    def observe(self, observation, reward, done):

        if np.random.rand() < 0.5:
            future = (not done) * self.Q2[observation, np.argmax(self.Q1[observation, :])]
            self.Q1[self.state, self.action] += self.alpha * (reward + self.gamma * future - self.Q1[self.state, self.action])
        else:
            future = (not done) * self.Q1[observation, np.argmax(self.Q2[observation, :])]
            self.Q2[self.state, self.action] += self.alpha * (reward + self.gamma * future - self.Q2[self.state, self.action])

        if not done:
            self.state = observation

    def show_values(self):
        print(self.Q1, self.Q2)
        import seaborn as sns
        p1 = sns.heatmap(self.Q1, annot=True, cmap='Greens')
        p1.set_title("Double Q-learning Q1")
        plt.show()
        p2 = sns.heatmap(self.Q2, annot=True, cmap='Greens')
        p2.set_title("Double Q-learning Q2")
        plt.show()

    def show_policy(self, env):
        policy = np.zeros([self.state_space, 1])
        for s in range(self.state_space):
            policy[s] = np.argmax(self.Q1[s, :] + self.Q2[s, :])
        if env == "FrozenLake-v1":
            policy = policy.reshape([4, 4])
        plot = sns.heatmap(policy, annot=True, cmap="Reds")
        plot.set_title("Double Q-learning greedy policy")
        plt.show()
