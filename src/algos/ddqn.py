import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

class Network(nn.Module):
    def __init__(self, state_size, action_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DDQN Agent
class DDQNAgent:
    def __init__(self, state_size, action_size, buffer, gamma, lr, tau):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = buffer.batch_size

        self.qnetwork_local = Network(state_size, action_size)
        self.qnetwork_target = Network(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.buffer = buffer
        self.update_target_network(self.qnetwork_local, self.qnetwork_target)
        
    def update_target_network(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(range(self.action_size))
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.qnetwork_local(state)).item()
        
    def step(self, state, action, reward, next_state, done, goal=None) -> float:
        self.buffer.add(state, action, reward, next_state, done, goal)
        if len(self.buffer) > self.batch_size:
            return self.learn()
        return 0
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 0        
        if self.buffer.name == "ReplayBuffer":
            states, actions, rewards, next_states, dones = self.buffer.sample()

            # Compute Q targets
            with torch.no_grad():
                next_actions = self.qnetwork_local(next_states).argmax(1)
                next_q_values = self.qnetwork_target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                q_targets = rewards + (self.gamma * next_q_values * (1 - dones))
                
            # Compute Q expected
            q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute loss
            loss = nn.MSELoss()(q_expected, q_targets)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.update_target_network(self.qnetwork_local, self.qnetwork_target)
            
            return loss.item()
        
        elif self.buffer.name == "PrioritizedReplayBuffer":
            states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample()
            
            # Compute Q targets
            with torch.no_grad():
                next_actions = self.qnetwork_local(next_states).argmax(1)
                next_q_values = self.qnetwork_target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                q_targets = rewards + (self.gamma * next_q_values * (1 - dones))
            
            # Compute Q expected
            q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute td error and loss
            td_error = q_expected - q_targets
            loss = (td_error ** 2 * weights).mean()
            
            # Backpropagate importance-weighted loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update priorities
            priorities = td_error.abs().detach().numpy().tolist()
            self.buffer.update_priorities(indices, priorities)
            
            # Soft update of target network
            self.update_target_network(self.qnetwork_local, self.qnetwork_target)
            
            return loss.item()
        
        elif self.buffer.name == "DHERReplayBuffer":
            # Sampling from DHER replay buffer
            states, actions, rewards, next_states, dones, goals = self.buffer.sample()

            # Convert dones from a boolean tensor to a float tensor for arithmetic operations
            dones = dones.float()

            # Calculate the Q targets for next states
            with torch.no_grad():
                q_targets_next = self.qnetwork_target(next_states).max(1)[0].detach()
                q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

            # Get expected Q values from the local model for the current states and actions
            q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute loss
            loss = nn.MSELoss()(q_expected, q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Soft update of target network
            self.update_target_network(self.qnetwork_local, self.qnetwork_target)

            return loss.item()
