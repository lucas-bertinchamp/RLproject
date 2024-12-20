import random
import torch
import numpy as np

class DHERReplayBuffer:
    def __init__(self, buffer_size, batch_size, max_failed_episodes=25):
        self.buffer = []
        self.failed_episodes = []
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.position = 0
        self.max_failed_episodes = max_failed_episodes  # Limit for failed episodes
        self.name = "DHERReplayBuffer"

    def add(self, state, action, reward, next_state, done, goal=None):
        """
        Add a transition to the buffer with an optional goal.

        If no goal is provided, use the next_state as a default goal.
        """
        if goal is None:
            goal = next_state
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done, goal))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done, goal)
        
        self.position = (self.position + 1) % self.buffer_size
    
    def add_episode(self, episode_transitions):
        """
        Add an entire episode to the buffer with DHER logic.
        """
        # Add original transitions to the buffer
        for transition in episode_transitions:
            self.add(*transition)
        
        # Check if episode is a failure and store it
        if not self._reached_goal(episode_transitions):
            if len(self.failed_episodes) >= self.max_failed_episodes:
                self.failed_episodes.pop(0)  # Remove oldest failed episode
            self.failed_episodes.append(episode_transitions)

            # Combine failed episodes with inverse simulation
            self._combine_with_failed_episodes(episode_transitions)
    
    def _reached_goal(self, episode_transitions):
        """
        Check if the final state in the episode achieves the goal.
        """
        final_state = episode_transitions[-1][3]  # Final next_state
        goal = episode_transitions[-1][5]        # Final goal
        return np.all(np.isclose(final_state, goal, atol=1e-3))

    def _combine_with_failed_episodes(self, new_episode):
        """
        Perform inverse simulation to generate imagined trajectories
        by combining failed episodes with the current new episode.
        """
        for failed_episode in self.failed_episodes:
            for t_new, transition_new in enumerate(new_episode):
                _, _, _, next_state_new, _, _ = transition_new  # Next state from new episode
                
                for t_failed, transition_failed in enumerate(failed_episode):
                    _, _, _, next_state_failed, _, goal_failed = transition_failed  # Failed goal
                    
                    # If goals are close, combine the trajectories
                    if np.linalg.norm(next_state_new - goal_failed) < 0.05:  # Check goal proximity
                        imagined_goal = goal_failed
                        for k in range(min(t_new, t_failed) + 1):
                            state, action, _, next_state, done, _ = new_episode[k]
                            reward = -np.linalg.norm(next_state - imagined_goal)  # Shaped reward
                            self.add(state, action, reward, next_state, done, imagined_goal)

    def sample(self):
        """
        Sample a batch of transitions from the buffer.
        """
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones, goals = zip(*batch)

        # Convert to tensors
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.int64),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
            torch.tensor(np.array(goals), dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)