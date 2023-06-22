"""
Author: Ameer H.
Description: This Python file will contain code from PyTorch's documentation about regarding Re-inforcement Learning Using PyTorch.
But instead of using the Mario Game, I will use some other game and program the code accordingly.

The code will follow similar style of the original MARIO RL Tutorial.

Elegant Robot's Code,
https://elegantrl.readthedocs.io/en/latest/tutorial/BipedalWalker-v3.html
"""

# ! Some facts about the LunarLander-v2
# ! Action Space
# * 1-Do Nothing, 2-Fire Left Rocket 3-Fire Main Engine 4-Fire Right Rocket

# ! Observation Space
# * 

from platform import release
import re
from tkinter import Frame
import torch
from torch import nn
from torchvision import transforms as T

from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque

import random, datetime, os, copy
import time
import matplotlib.pyplot as plt
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# * Let's initialize the LunarLander Environment
if gym.__version__ < '0.26':
    # env = gym.make("LunarLander-v2", render_mode="human", new_step_api=True, continuous=True)
    env = gym.make("LunarLander-v2", new_step_api=True, continuous=True)
    # env = gym.make("BipedalWalker-v3", render_mode='human')
else:
    env = gym.make("LunarLander-v2", render_mode="rgb", apply_api_compatibility=True, continuous=True)

# ! Put the action space overe here
# ! Example Action -> np.array([main, lateral])
# ! if main < 0 - main thruster is turned off
# ! 0 <= main <= 1.
# ! lateral - Has two possibilities,
# ! if -1.0 < lateral < -0.5 -> Left booster will fire
# ! if 0.5 < lateral < 1 -> Right booster will fire.

# env.action_space = Box
# ! An example action space.
action_X = env.action_space.sample()
print(f"Action Space: {action_X.shape}")        # Action Space -> 0 in Mario RL

# Maybe we don't need to define the action space here and we can simply,
# put random values in the act!


env.reset()

# ? This is basically a random action from the action space
next_state, reward, done, _, info = env.step(action=np.array([0.5, -0.5]))
print(f"State Shape: {next_state.shape}. \n Reward: {reward}, \n Done: {done} \n Info: {info}")
print(f'Next State {next_state}')
# ! Original in Mario -> (240, 256, 3)
# ! State shape in Lander -> (8,)
# ! Channels: 8 & Channel Type class <'int'>
# ! Output Channels: 2 --> Which is equal to ([main, lateral])

# * %% Preprocess the Environment
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        # ? Return only every skip -th frame
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        # ? Repeat action and sum reward
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate all rewards and repeat the action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward

            if done:
                break
            return obs, total_reward, done, info
    
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def permute_orientation(self, observation):
        # [H, W, C] to [C, H, W] tensor
        # observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape(shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

# ! Now we apply WRAPPERS to environment
# env = SkipFrame(env, skip=4)
# env = GrayScaleObservation(env)
# env = ResizeObservation(env, shape=8)

# if gym.__version__ < '0.26':
#     env = FrameStack(env, num_stack=4, new_step_api=True)
# else:
#     env = FrameStack(env, num_stack=4)
# %%
# ! Let's Create the Agent

class Lander:
    def __init__(self, state_dim, action_dim, save_dir):

        self.state_dim = state_dim      # ? should be 8
        self.action_dim = action_dim    # ? should be 2
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        
        
        # print(f"Type of state: {type(self.state_dim)}")         # State in Mario here -> (4, 84, 84) class <'tuple'>
        # print(f"State: {self.state_dim}")
        # print(f"Action: {self.action_dim}")                     # Action in Mario here -> 2 class <'int'>

        # self.net = LanderNet(self.state_dim, self.action_dim)
        self.net = LanderNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5      # das

        self.memory = deque(maxlen=100000)

        self.batch_size = 8 #32
        self.gamma = 0.9
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4
        self.learn_every = 3
        self.sync_every = 1e4

    
    def act(self, state):
        """Action

        Args:
            state (LazyFrame): A single observation of the current state.

        Returns:
            action_idx (int): Best action based on Explore or Exploit
        """

        # TODO This function is currently returning a single index List [0] or [1]
        # TODO This was the action space for Mario's running and not for Lunar Lander.
        # TODO The action space of the lunar lander is of the following nature.
        # TODO --> np.array([main_engine, lateral_engines])
        # EXPLORE
        x = np.random.rand()
        # print(f'The Random: {x} & the Exploration: {self.exploration_rate}')
        # It is not the exploration rate's fault that my program is stopping.
        if x < self.exploration_rate:
            # action_idx = np.random.randint(self.action_dim)
            # ! maybe it will be,
            action_idx = env.action_space.sample()  # ! Yes it is!
        
        #EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()

            state = torch.tensor(state, device=self.device).unsqueeze(0)
             # ? UNSQUEEZE -> Returns a new tensor with a dimension of size one inserted at the specified position.
            # print("EXPLOIT: {state}")
            action_values = self.net(state, model='online')
            # print(f"Action Values: {action_values}")
            action_idx = torch.argmax(action_values, axis=1).item()

        
        # decrease the exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # print(self.exploration_rate)

        # increment step
        self.curr_step += 1
        # print(f'The Chosen ONE!: {action_idx}')         # ! Example Action State -> [ 0.16972001 -0.9608854 ]
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        """The landers memory

        Args:
            state
            next_state
            action (int)
            reward (float)
            done (bool)
        """

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)

        # action = first_if_tuple(action).__array__()
        # reward = first_if_tuple(reward).__array__()
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        """Sample experiences from memory by random samples

        Returns:
            Return a batch
        """

        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        """
        Putting it all together
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        # ! Sample from MEMORY
        state, next_state, action, reward, done = self.recall()

        # ! GET TD Estimate
        td_est = self.td_estimate(state, action)

        # ! GET TD Target
        td_trgt = self.td_target(reward, next_state, done)

        # ! Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_trgt)

        return (td_est.mean().item, loss)

    # ! TD Estimate & TD Learning
    # ! We are basically trying to approximate Q(s,a) with a neural network.
    # ! To do this we have to calculate targets using Bellman's equations.

    # * Disable gradient to avoid backprop
    # ? TD ESTIMATE & TD LEARNING
    # ? Two values are involved in learning
    # ? TD ESTIMATE - Optimal Q* for a given state 's' is TDe = Q*_online(s, a)
    # ? TD TARGET - Aggregation of current reward and estimated Q* in the next state s' is
    # ?                         a' = argmaxQonline(s', a)
    # ?                         TD_target = reward + discount x Q*_target(s', a')
    # ? Since we don't know what next action a' will be, we use the action a' that maximizes Qonline in the next state s'

    def td_estimate(self, state, action):
        x =  [np.arange(0, self.batch_size), action]          # ! Create a row of batch size so you can multiply it with action which has the same number of rows as batch size.
        # print(f'X: {x} and Action {action} and Action Type: {type(action)} and Action Shape: {action.shape}')        # The action space in MARIO tutorial is limited to [0, 1] While ours is a bit different [main, lateral].
        # ! Online learning -> Approach used in Machine Learning that takes in sample of real time data one observation at a time.
        
        current_Q = self.net(state, model='online')[
            np.arange(0, self.batch_size*2).reshape((8,2)), action.long()               # Problem here is that the first 
            ]   # Q_online
        # X: [0 1 2 3 4 5 6 7] and 
        # Action tensor
        # ([[-0.8828,  0.1326],
        # [ 0.7974, -0.5608],
        # [-0.4290, -0.4602],
        # [ 0.6359,  0.0105],
        # [ 0.1477, -0.6896],
        # [-0.4523,  0.5478],
        # [ 0.9237, -0.9010],
        # [-0.8544,  0.7527]])
        # print(f'Current Q: {current_Q} and x: {x}')
        # print(f'Current Q -> {current_Q}')
        return current_Q
    # ! Important to note that target network's parameters are not trained, but they are periodically synchronized with the paramters of
    # ! the main Q-network. The idea is that using the target network's Q values to train the main Q-network will improve the stability
    # ! of the training.
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        x =  [np.arange(0, self.batch_size), action]
        # print(f'TD_Target Normal Action--> X: {x} and Action {action} and Action Type: {type(action)}')
        next_state_Q = self.net(next_state, model='online')
        # print(f'Next State Q: {next_state_Q} and and Size {next_state_Q.shape}')
        best_action = torch.argmax(next_state_Q, axis=1)
        print(f'TD_Target --> X: {x} and Action {best_action} and Action Type: {type(best_action)}')
        next_Q = self.net(next_state, model='target')[
            np.arange(0, self.batch_size*2).reshape(8,2), best_action[:16].reshape(8, 2)
        ]
        # TD_target = reward + discount x Q*_target(s', a')
        # print(f'NEXT Q --> {next_Q}')


        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    # ! Update the model
    def update_Q_online(self, td_estimate, td_target):
        print(f'Size TDe: {td_estimate.shape}, Size TDt: {td_target.shape}')
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    # ! Save the checkpoint
    def save(self):
        save_path = (
            self.save_dir / f"lander_net_{int(self.curr_step // self.save_every)}"
        )

        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Lander Net save to {save_path} at step {self.curr_step}")

# ! FINALLY THE LanderNet DQNN Algorithm
class LanderNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c = input_dim
        out = output_dim[0]
        print("Using CONV")
        print(f'Channels: {c} & Channel Type {type(c)}')        # Channels: 8 & Channel Type class <'int'>
        print(f'Output Channel: {output_dim[0]}')               # Output Channels: 2
        # if h != 84:
        #     raise ValueError
        # if w != 84:
        #     raise ValueError

        # !! YE HAI ONLINE

        self.online = nn.Sequential(
            # nn.Conv2d(in_channels=c, out_channels=32, kernel_size=4, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1),
            # nn.ReLU(),
            # nn.Flatten(),
            # nn.Linear(3136, 512),
            # nn.ReLU(),
            # nn.Linear(512, output_dim[0]),
            nn.Conv1d(in_channels=c, out_channels=32, kernel_size=4),  # ? in = 8, 
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        # * The forward pass
        if model == 'online':
            # print(f'Input {input} has len: {type(input)}') # ! Input tensor([[-0.3194,  1.1934, -0.7491, -0.5035,  0.4231,  0.0699,  0.0000,  0.0000]]) has len: <class 'torch.Tensor'>
            return self.online(input)
        elif model == 'target':
            return self.target(input)

# %%
# ! ------------------------
# ! LOGGING
# ! ------------------------
class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode ' :>8}{'Step ' :>8}{'Epsilon' :>10}{'MeanReward' :>15}"
                f"{'MeanLength ':>15}{'MeanLoss ':>15}{'MeanQValue ':>15}"
                f"{'TimeDelta ':>15}{'Time ':>20}\n" 
            )

        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History Metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call ti record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # current episode metric
        self.init_episode()

        # TIMING
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1

        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        # Mark end of episode
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)

        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - ",
            f"Step {step} - ",
            f"Epsilon {epsilon} - ",
            f"Mean Reward {mean_ep_reward} - ",
            f"Mean Length {mean_ep_length} - ",
            f"Mean Loss {mean_ep_loss} - ",
            f"Mean Q Value {mean_ep_q} - ",
            f"Time Delta {time_since_last_record} - ",
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d} {step:8d} {epsilon:10.3f}"
                f"{mean_ep_reward:15.3f} {mean_ep_length:15.3f} {mean_ep_loss:15.3f} {mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))        # ? GET THE ATTRIBUTE NAMED metric.  getattr(x, 'y' simply means x.y but I guess some difference in classes)
            plt.clf()
# %%
# ! ~><~~><~~><~~><~~><~~><~~><~~><~~><~~>
# ? TIME TO PLAY
# ! ~><~~><~~><~~><~~><~~><~~><~~><~~><~~>

use_cuda = torch.cuda.is_available()
print(f"Are we USING CUDA? \n{use_cuda}")
print()
save_dir = Path("checkpoints") / datetime.datetime.now().strftime('%Y-%m-%d T%H-%M-%S')
# save_dir = os.path.join("checkpoints", datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))
save_dir.mkdir(parents=True)
# if os.path.exists(save_dir):

lander = Lander(state_dim=8, action_dim=env.action_space.shape, save_dir=save_dir)
logger = MetricLogger(save_dir=save_dir)

episodes = 100
for e in range(episodes):
    state = env.reset()

    # * PLAY THE GAME
    while True:
        # * Run agent on the state
        # print("ACT")
        action = lander.act(state)
        # print(action)

        # * Agent performs the action
        next_state, reward, done, _, info = env.step(action)

        # * Remember
        # print("REMEMBER")
        lander.cache(state, next_state, action, reward, done)

        # * Learn
        # print("LEARN")
        q, loss = lander.learn()
        if (q is not None and loss is not None):
            print(q)
            print(loss)

        # * Log
        # print("LOG")
        logger.log_step(reward, loss, q)

        # * Update step
        # print("UPDATE")
        state = next_state

        # * Check if end of GAME
        if done:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=lander.exploration_rate, step=lander.curr_step)





        

    
