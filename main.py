import gym
from src.algos.dqn import DQNAgent
from src.buffers.replay_buffer import ReplayBuffer
from src.trainer import train
from src.utils import *
import argparse
import random
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment to train")
    parser.add_argument("--agent", type=str, default="dqn", help="Agent to use")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument("--buffer", type=str, default="ReplayBuffer", help="Buffer to use")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Size of the buffer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--n_episodes", type=int, default=250, help="Number of episodes to train")
    parser.add_argument("--name", type=str, default="agent", help="Name of the model")
    return parser


if __name__ == "__main__":

    agents = {
        "dqn": DQNAgent
    }
    
    buffers = {
        "ReplayBuffer": ReplayBuffer
    }
    
    parser = get_parser()
    args = parser.parse_args()
    print(f"Arguments: {args}")

    # Initialize environment
    env = gym.make(args.env)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    
    # Initialize buffer and agent
    buffer = buffers[args.buffer](buffer_size=args.buffer_size, 
                                  batch_size=args.batch_size)
    agent = agents[args.agent](state_size=state_size, 
                               action_size=action_size, 
                               buffer=buffer, 
                               gamma=0.99, 
                               lr=0.001, 
                               tau=0.001)

    # Train agent
    scores, training_loss = train(
        env=env,
        agent=agent,
        n_episodes=args.n_episodes,
        max_t=1000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )

    # Save pipeline
    print("Training complete.")
    args.name = create_folders(args.name)
    plot_training(scores, training_loss, model_name=args.name)
    save_model(agent, args.name)
    save_params(vars(args), args.name)
    
    # visualize_agent(env, agent)