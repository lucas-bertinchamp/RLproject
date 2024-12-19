import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from src.algos.dqn import DQNAgent
import sys
import shutil
import gym
from gym.wrappers import RecordVideo
from pathlib import Path
import subprocess

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
            
    np.save(f"models/{model_name}/plots/scores.npy", scores)
    np.save(f"models/{model_name}/plots/loss.npy", training_loss)
        
        
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
    
def create_video(env, agent, video_folder='videos', n_episodes=25):
    # Clean up video folder
    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)
    os.makedirs(video_folder)

    # Use RecordVideo wrapper to save video frames
    env = RecordVideo(env, video_folder)

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.select_action(state, epsilon=0.0)  # Greedy policy (epsilon=0)
            state, reward, done, _ = env.step(action)
            score += reward

    env.close()
    
def del_temp():
    if os.path.exists("models/temp"):
        shutil.rmtree("models/temp")
    
def concat_videos(model_name):

    video_folder = f"models/temp/"
    path = Path(video_folder)
    # Find all videos in the folder and sort them
    videos_path = [f for f in path.rglob("*.mp4")]
    order_dict = {}
    for video_path in videos_path:
        episode = video_path.split("-")[-1]
        episode = video_path.split(".")[0]
        order_dict[int(episode)] = video_path
        
    videos_path = [order_dict[k] for k in sorted(order_dict.keys())]
    
    # Concat
    list_file = "video_list.txt"
    with open(list_file, "w") as f:
        for video in videos_path:
            f.write(f"file '{video}'\n")

    output_file = f"models/{model_name}/video.mp4"
    
    # Commande FFmpeg pour concaténer les vidéos
    command = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",  # Copie les flux sans réencodage
        output_file
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Vidéo concaténée avec succès : {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la concaténation : {e}")
    finally:
        # Supprimez le fichier temporaire
        Path(list_file).unlink()