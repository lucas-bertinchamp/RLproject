import random
import torch

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = []
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        self.position = 0
        
        self.name = "ReplayBuffer"
        
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
        self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)