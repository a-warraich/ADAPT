import numpy as np

class Logger:
    def __init__(self):
        self.steps = 0
        self.total_steps = 0
        self.new_line_every = 25000
        
        self.cumulative_reward = 0
        self.episode_rewards = []
        self.episode_rewards_ma = 0
        
        self.current_episode_length = 0
        self.episode_lengths = []
        self.episode_lengths_ma = 0

        self.cumulative_iae = 0
        self.episode_iaes = []
        self.episode_iaes_ma = 0

    def log(self, reward, error, done):

        self.cumulative_reward += reward
        self.cumulative_iae += abs(error)
        self.current_episode_length += 1
        self.steps += 1

        if done:
            self.episode_rewards.append(self.cumulative_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_iaes.append(self.cumulative_iae)

            self.episode_rewards_ma = np.mean(self.episode_rewards[-50:])
            self.episode_lengths_ma = np.mean(self.episode_lengths[-50:])
            self.episode_iaes_ma = np.mean(self.episode_iaes[-50:])

            self.cumulative_reward = 0
            self.current_episode_length = 0
            self.cumulative_iae = 0

    def print_logs(self):
        end_char = "\n" if self.steps % self.new_line_every == 0 else "\r"
        print(
            f"Step: {self.steps}/{self.total_steps} | "
            f"Avg Reward: {self.episode_rewards_ma:.4f} | "
            f"Avg Steps: {self.episode_lengths_ma:.2f} | "
            f"Avg IAE: {self.episode_iaes_ma:.4f}",
            end=end_char
        )
