import numpy as np
from src.utils import create_video

def train(env, agent, n_episodes, max_t, epsilon_start, epsilon_end, epsilon_decay, name=None):
    scores = []
    training_loss = []
    epsilon = epsilon_start
    env_video = None

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0
        episode_loss = 0
        episode_transitions = []  # Store the transitions for the full episode
        for t in range(max_t):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            episode_loss += agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
            # Add the entire episode to the HER buffer (with hindsight goals)
        if hasattr(agent.buffer, "add_episode"):
            agent.buffer.add_episode(episode_transitions)
        else:
            for transition in episode_transitions:
                agent.buffer.add(*transition)  # Default behavior if no HER
            
            # Trigger learning
        episode_loss += agent.learn()
        training_loss.append(episode_loss / max(1, t))

        scores.append(total_reward)
        epsilon = max(epsilon_end, epsilon_decay * epsilon)


        if episode % 10 == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores[-100:]):.2f}")
            create_video(env, agent, video_folder=f'models/temp/{name}_videos/{episode}', n_episodes=1)

    return scores, training_loss