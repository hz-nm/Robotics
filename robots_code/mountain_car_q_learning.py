# Original Article -> https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
# Author -> Moustafa Alzantot


# Q-Learning Example using OpenAI GYM Mountain Car Environment

# Get all the environments here,
# import gym
# envs = gym.envs.registry.all()
# print(envs)

import numpy as np
import gym
from gym import wrappers

n_states = 40           # no of states in the environment
iter_max = 10000        # max iterations

initial_lr = 1.0        # initial learning rate
min_lr = 0.003          # min learning rate to change
gamma = 1.0             # discount
t_max = 10000          # max number of tries in a single simulation.
eps = 0.02              # probability?

def obs_to_state(env, obs):
    """Maps an observation to state

    Args:
        env : Environment from GYM
        obs : observation from running an episode

    Returns:
        Mapped states
    """
    env_low = env.observation_space.low
    env_high = env.observation_space.high

    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])

    return a, b

def run_episode(env, policy=None, render=False):
    """Runs an episode of the simulation with or without the given policy

    Args:
        env (): Environment
        policy (optional): Policy for RL Algorithm. Defaults to None.
        render (bool, optional): Render the animation or not?. Defaults to False.

    Returns:
        Reward: Reward acheived by the simulation in the current environment.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0

    for _ in range(t_max):      #tmax instead of 10000
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a, b = obs_to_state(env, obs)
            action = policy[a][b]
        
        obs, reward, done, _ = env.step(action)
        total_reward += (gamma ** step_idx) * reward
        step_idx += 1

        if done:
            break
    return total_reward

if __name__ == "__main__":
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    seed = 0
    np.random.seed(200)

    print("Let's use Q-Learning")
    q_table = np.zeros((n_states, n_states, 3))

    for i in range(iter_max):
        obs = env.reset(seed=seed)
        total_reward = 0

        # eta: learning rate is decreased at each step
        eta = max(min_lr, initial_lr * (0.85 ** (i/100)))       # pronounced eeta
        for j in range(t_max):
            a, b = obs_to_state(env, obs)
            if np.random.uniform(0, 1) < eps:
                # pick a random action if probability is less than threshold set above
                action = np.random.choice(env.action_space.n)
            else:
                # pick an action according to the current estimates of Q_Values.
                logits = q_table[a][b]
                logits_exp = np.exp(logits)
                probs = logits_exp / np.sum(logits_exp)

                action = np.random.choice(env.action_space.n, p=probs)

            obs, reward, done, _ = env.step(action)

            total_reward += (gamma ** j) * reward

            # update q table
            a_, b_ = obs_to_state(env, obs)
            q_table[a][b][action] = q_table[a][b][action] + eta * (reward + gamma * np.max(q_table[a_][b_]) - q_table[a][b][action]) # updating what was there before.
            if done:
                break

        if i % 100 == 0:
            # print after every 100 iterations
            print(f'Iteration # {i+1}, -- Total reward = {total_reward}')
    
    solution_policy = np.argmax(q_table, axis=2)

    solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]

    print("Average score of solution = ", np.mean(solution_policy_scores))

    # and now we will animate
    goto_next = input('Would you like to run with the selected policy? (Y/n)').lower()
    if goto_next == 'y':
        env_2 = gym.make(env_name, render_mode='human')
        env_2.reset(seed=seed)
        run_episode(env=env_2, policy=solution_policy, render=True)
    else:
        pass

    print('Test and Simulation completed!')