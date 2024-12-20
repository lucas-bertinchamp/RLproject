import random
import numpy as np
import torch
from collections import deque

class DHERReplayBuffer:
    def __init__(self, buffer_size, batch_size, env, gamma, alpha=0.6, k=4):
        """
        Initialize a Dynamic Hindsight Experience Replay Buffer.

        Args:
            buffer_size (int): Maximum number of transitions to store in the buffer.
            batch_size (int): Number of transitions to sample from the buffer.
            get_q_value_func (callable): Function to compute Q-values for states and actions.
            gamma (float): Discount factor for future rewards.
            alpha (float): Determines the extent of prioritization based on TD Errors.
        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.position = 0
        self.name = "DHERReplayBuffer"
        self.k = k

    def add(self, state, action, reward, next_state, done, goal=None):
        
        goal_pos, goal_vel = goal
        self.buffer.append((state, action, reward, next_state, done, np.array([goal_pos])))
        
        for _ in range(self.k):
            dynamic_goal = self.generate_dynamic_goal(next_state)
            # Recalculate reward based on the new dynamic goal
            reward_dynamic = self.compute_reward(next_state, dynamic_goal)
            done_dynamic = self.is_goal_reached(next_state, dynamic_goal)
            # Store the hindsight transition
            self.buffer.append((state, action, reward_dynamic, next_state, done_dynamic, dynamic_goal))

    def generate_dynamic_goal(self, state):
        """
        Dynamically generates a goal based on the agent's needs.
        """
        # Example: Select a random position in the current range
        position, _ = state
        dynamic_goal_position = np.clip(position + np.random.uniform(-0.1, 0.1), self.env.min_position, self.env.max_position)
        return np.array([dynamic_goal_position])
    
    def compute_reward(self, state, goal):
        """
        Computes the reward based on the dynamic goal.
        """
        position, _ = state
        return 0 if np.abs(position - goal[0]) < 0.05 else -1  # Small tolerance for reaching the goal

    def is_goal_reached(self, state, goal):
        """
        Checks if the dynamic goal has been reached.
        """
        position, _ = state
        return np.abs(position - goal[0]) < 0.05

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones, goals = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(goals, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)
