import numpy as np
import torch

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha=0.6):
        """
        Initialize a Prioritized Replay Buffer.

        Args:
            buffer_size : Maximum number of transitions to store in the buffer.
            alpha : Determines the level of prioritization. 0 corresponds to uniform sampling.
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        
        self.name = "PrioritizedReplayBuffer"

    def add(self, state, action, reward, next_state, done, goal=None):
        """
        Add a new experience to the buffer.
        """
        max_priority = max(self.priorities, default=1.0)  # Assign max priority to new transitions

        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.buffer_size

    def sample(self, beta=0.4):
        """
        Sample a batch of experiences from the buffer.
        """
        if len(self.buffer) == 0:
            raise ValueError("The buffer is empty")
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Indices based on the probabilities
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # Importance-sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            indices
        )

    def update_priorities(self, indices, priorities):
        """
        Update the priorities of sampled transitions.
        """
        for idx, priority in zip(indices, priorities):
            # Update the priority of the transition at the given index
            self.priorities[idx] = max(priority, 1e-5) # Ensure no priority is too small (issue)
            
    def __len__(self):
        return len(self.buffer)
