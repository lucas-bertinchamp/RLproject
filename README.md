# INF8250AE - Reinforcement Learning Project Code

This repository contains the code for the project of the course INF8250AE - Reinforcement Learning at Polytechnique Montréal.

Project members:

- Lucas Bertinchamp - 2312324
- Hélène Genet - 2235745
- Niels Mainville - 2080618

## Project Description

The goal of this project is to implement and compare the performance of different buffers in the context of reinforcement learning. We will implement the following buffers:

- Replay Buffer
- Prioritized Experience Replay (PER)
- Hindsight Experience Replay (HER)
- Diverse Hindsight Experience Replay (DHER)

## Training

To train the agent, run the following command:

```bash
python main.py --env <env_name> --agent <agent_name> --seed <seed> --buffer <buffer_name> --buffer_size <buffer_size> --batch_size <batch_size> --n_episodes <n_episodes> --name <name>
```

For example, to train the agent on the CartPole environment with the DQN agent and the ReplayBuffer buffer on 1000 episodes, run the following command:

```bash
python main.py --env CartPole-v1 --agent dqn --buffer ReplayBuffer --n_episodes 1000 --name cartpole_repbuff_dqn
```

Notice that all arguments are optional and have default values. The default values are the following:

- `env`: `CartPole-v1`
- `agent`: `dqn`
- `seed`: `0`
- `buffer`: `ReplayBuffer`
- `buffer_size`: `10000`
- `batch_size`: `64`
- `n_episodes`: `250`
- `name`: `agent`

After training, the results will be saved in the `models` folder which will contain the following files:

- `model_name/checkpoints/`: folder containing the model checkpoints
- `model_name/plots/`: folder containing the training plots
- `model_name/command.txt`: file containing the command used to train the model
- `model_name/params.json`: file containing the parameters used to train the model

## Creating a new buffer

Here are the steps to create and use a new buffer:

- Add a new buffer class in the `src/buffers` folder. The class should implement the `add` and `sample` methods. The `__init__` method should take the following arguments:
  - `buffer_size`: the maximum number of elements in the buffer
  - `batch_size`: the number of elements to sample at each call to the `sample` method
- Add an attribute `name` to the buffer class. This attribute should be a string containing the name of the buffer.
- In the `src/algos/dqn.py` file, add an elif statement in the `learn` method to handle the new buffer.
- Import the buffer class in the `main.py` file.
- Add the buffer in the `buffers` dictionnary in the `main.py` file.
- You can now use the buffer as an argument of `--buffer` in the command line.
