import numpy as np

class MultiSystemPIDEnv:
    def __init__(self, desired_state=1.0, max_steps=200):
        self.desired_state = desired_state
        self.max_steps = max_steps
        self.step_count = 0
        self.sample_random_system()

    def sample_random_system(self):
        self.mass = np.random.uniform(1.0, 5.0)
        self.friction = np.random.uniform(0.1, 1.0)
        self.time_constant = np.random.uniform(0.5, 2.0)

        self.motor_speed = 0
        self.integral = 0
        self.prev_error = 0
        self.dt = 0.1

    def reset(self):
        self.step_count = 0
        self.sample_random_system()
        self.motor_speed = 0
        self.integral = 0
        self.prev_error = 0
        initial_state = self._get_state()
        return initial_state

    def step(self, action):
        self.step_count += 1
        Kp, Ki, Kd = action

        error = self.desired_state - self.motor_speed

        P_out = Kp * error
        self.integral += error * self.dt
        I_out = Ki * self.integral
        derivative = (error - self.prev_error) / self.dt
        D_out = Kd * derivative

        control_output = P_out + I_out + D_out

        acceleration = (control_output - self.friction * self.motor_speed) / self.mass
        self.motor_speed += acceleration * self.dt / self.time_constant

        self.prev_error = error

        next_state = self._get_state()
        # Original negative reward function (works well with proper hyperparameters)
        reward = -abs(error)  # Range: -∞ to 0, less negative = better performance
        
        # Alternative reward functions (uncomment to use):
        # reward = 1.0 / (1.0 + abs(error))  # Positive reward: range: 0 to 1, higher is better
        # reward = np.exp(-abs(error))  # Exponential decay: range: 0 to 1
        # reward = 1.0 - abs(error) / self.desired_state  # Linear: range: 0 to 1
        # reward = 1.0 / (1.0 + abs(error)**2)  # Squared error penalty
        
        # Enhanced reward with multiple components (uncomment to use):
        # error_reward = 1.0 / (1.0 + abs(error))
        # stability_reward = 1.0 / (1.0 + abs(derivative))  # Penalize oscillations
        # settling_reward = 1.0 if abs(error) < 0.1 else 0.0  # Bonus for good settling
        # reward = error_reward + 0.1 * stability_reward + 0.2 * settling_reward
        
        done = self.step_count >= self.max_steps

        return next_state, reward, done, {'error': error}

    def _get_state(self):
        error = self.desired_state - self.motor_speed
        return np.array([self.motor_speed, error, self.mass, self.friction, self.time_constant])
