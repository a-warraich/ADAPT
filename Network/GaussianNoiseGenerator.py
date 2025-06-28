import numpy as np

class GaussianNoiseGenerator:
    def __init__(self, action_dim, sigma=0.1, noise_clip=None, action_low=None, action_high=None):
        self.sigma = sigma
        self.noise_clip = noise_clip 
        self.action_dim = action_dim

        self.action_low = np.array(action_low) if action_low is not None else None
        self.action_high = np.array(action_high) if action_high is not None else None

    def sample(self):
        noise = self.sigma * np.random.randn(self.action_dim)
        
        if self.noise_clip is not None:
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
        
        return noise

    def apply(self, action):
        noisy_action = action + self.sample()

        if self.action_low is not None and self.action_high is not None:
            noisy_action = np.clip(noisy_action, self.action_low, self.action_high)

        return noisy_action
