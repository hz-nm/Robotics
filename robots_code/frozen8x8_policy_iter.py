# https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
# Using Policy Iteration to Solve FrozenLake8x8 Problem from OpenAI Gym

import numpy as np
import gym
from gym import wrappers

def run_episode(env, policy, gamma=1.0, render=False):

    obs = env.reset()
    total_reward = 0
    step_idx = 0
    render = True

    while True:
        if render:
            env.render()

        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx) * reward
        step_idx += 1

        if done:
            break
    return total_reward

def evaluate_policy(env, policy, gamma=1.0, n=100):
    scores = [
        run_episode(env, policy, gamma, False) for _ in range(n)
    ]

    return np.mean(scores)

def extract_policy(env, v, gamma = 1.0):
    env_nS = env.nrow * env.ncol        # possible states
    env_nA = 4                          # possible actions -> UP, DOWN, LEFT, RIGHT

    # policy = np.zeros((env.nrow, env.ncol))           # develop a sample policy
    policy = np.zeros(env_nS)

    for s in range(env_nS):
        q_sa = np.zeros(env_nA)
        for a in range(env_nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])
        policy[s] = np.argmax(q_sa)

    return policy

def compute_policy_v(env, policy, gamma=1.0):

    env_nS = env.nrow * env.ncol
    v = np.zeros(env_nS)

    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env_nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break

    return v

def policy_iteration(env, gamma=1.0):

    env_nA = 4
    policy = np.random.choice(env_nA, size=(env.nrow * env.ncol))        # initialize a random policy
    max_iterations = 200000
    gamma = 1.0

    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)

        if (np.all(policy == new_policy)):
            print('Policy-Iteration converged at step {}.'.format(i+1))
            break
        policy = new_policy

    return policy


if __name__ == "__main__":
    env_name = 'FrozenLake8x8-v1'
    env = gym.make(env_name)

    optimal_policy = policy_iteration(env, gamma=1.0)
    scores = evaluate_policy(env, optimal_policy, gamma=1.0)
    print('Average Scores: {}'.format(np.mean(scores)))




