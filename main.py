import gym
from src.algos.dqn import DQNAgent
from src.algos.ddqn import DDQNAgent
from src.buffers.replay_buffer import ReplayBuffer
from src.buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from src.buffers.HER_buffer import HERReplayBuffer
from src.buffers.DHER_buffer import DHERReplayBuffer
from src.trainer import train
from src.utils import *
import argparse
import random
import numpy as np
import warnings

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
    
    warnings.filterwarnings("ignore")   
    del_temp()

    agents = {
        "dqn": DQNAgent,
        "ddqn": DDQNAgent
    }
    
    buffers = {
        "ReplayBuffer": ReplayBuffer,
        "PrioritizedReplayBuffer": PrioritizedReplayBuffer,
        "HERReplayBuffer": HERReplayBuffer,
        "DHERReplayBuffer": DHERReplayBuffer
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
    
    if args.buffer == "DHERReplayBuffer":
        buffer = DHERReplayBuffer(
            buffer_size=args.buffer_size, 
            batch_size=args.batch_size,
            env=env,
            gamma=0.99)
        agent = DQNAgent(state_size, action_size, buffer, 0.99, 0.001, 0.001)
        
    elif args.buffer == "HERReplayBuffer":
        buffer = HERReplayBuffer(
            buffer_size=args.buffer_size, 
            batch_size=args.batch_size,
        )
        agent = DQNAgent(state_size, action_size, buffer, 0.99, 0.001, 0.001)
    else:
        buffer = buffers[args.buffer](buffer_size=args.buffer_size, batch_size=args.batch_size)
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
        epsilon_end=0.05,
        epsilon_decay=0.995,
        name = args.name
    )

    # Save pipeline
    print("Training complete.")
    args.name = create_folders(args.name)
    plot_training(scores, training_loss, model_name=args.name)
    save_model(agent, args.name)
    save_params(vars(args), args.name)
    concat_videos(args.name)
    del_temp()
    
    # visualize_agent(env, agent)