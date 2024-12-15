import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from src.algos.dqn import DQNAgent
import sys

def create_folders(model_name):
    os.makedirs("models", exist_ok=True)
    
    # If model already exists, add a number to the end
    if os.path.exists(f"models/{model_name}"):
        i = 1
        while os.path.exists(f"models/{model_name}_{i}"):
            i += 1
        model_name = f"{model_name}_{i}"
    
    # Create folders
    if model_name is not None:
        os.makedirs(f"models/{model_name}", exist_ok=True)
        os.makedirs(f"models/{model_name}/checkpoints", exist_ok=True)
        os.makedirs(f"models/{model_name}/plots", exist_ok=True)
        
    return model_name

def plot_training(scores, training_loss, model_name=None):
    
    data = [scores, training_loss]
    
    for i, d in enumerate(data):
        plt.figure()
        plt.plot(d)
        plt.title("Training Scores" if i == 0 else "Training Loss")
        plt.xlabel("Episode")
        plt.ylabel("Score" if i == 0 else "Loss")
        
        if model_name is not None:
            plt.savefig(f"models/{model_name}/plots/{'scores' if i == 0 else 'loss'}.png")
            
    with open(f"models/{model_name}/plots/scores.txt", "w") as f:
        f.write(str(scores))
        
    with open(f"models/{model_name}/plots/loss.txt", "w") as f:
        f.write(str(training_loss))
        
        
def save_model(agent, model_name):
    if type(agent) == DQNAgent:
        torch.save(agent.qnetwork_local.state_dict(), f"models/{model_name}/checkpoints/qnetwork_local.pth")
        torch.save(agent.qnetwork_target.state_dict(), f"models/{model_name}/checkpoints/qnetwork_target.pth")

def load_model(agent, model_name):
    if type(agent) == DQNAgent:
        agent.qnetwork_local.load_state_dict(torch.load(f"models/{model_name}/checkpoints/qnetwork_local.pth"))
        agent.qnetwork_target.load_state_dict(torch.load(f"models/{model_name}/checkpoints/qnetwork_target.pth"))
        
    return agent

def save_params(args, model_name):
    with open(f"models/{model_name}/params.json", "w") as f:
        f.write(str(args))
    f.close()
    
    with open(f"models/{model_name}/command.txt", "w") as f:
        f.write(" ".join(["python"] + sys.argv))
    
def visualize_agent(env, agent):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = agent.select_action(state, epsilon=0.0)  # Exploitation only
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        
    env.close()
    print(f"Total Reward: {total_reward}")