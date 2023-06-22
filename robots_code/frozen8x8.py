# Link -> https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
# Author -> Moustafa Alzantot

import numpy as np
import gym
from gym import wrappers


def run_episode(env, policy, gamma=0.99, render=True):
    """Evaluates Policy by using it to run an episode and finding its total reward.

    Args:
        env : gym environment
        policy : the policy to be used
        gamma (float, optional): discount factor. Defaults to 1.0.
        render (bool, optional): render the animation - Boolean. Defaults to False.

    returns:
        total reward: real value of the the total reward received by the agent under policy
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0

    while True:
        if render:
            # env.render()
            pass

        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += ((gamma ** step_idx) * reward)
        step_idx += 1
        if done:
            break
    
    return total_reward

def evaluate_policy(env, policy, gamma=0.99, n=100):
    """Evaluate a policy by runnning it n times.

    Args:
        env : gym environment
        policy : policy to be evaluated
        gamma (float, optional): discount. Defaults to 1.0.
        n (int, optional): iterations. Defaults to 100.

    Returns:
        average of total reward.
    """
    scores = [
        run_episode(env, policy, gamma=gamma, render=True)
        for _ in range(n)
    ]

    return np.mean(scores)

def extract_policy(env, v, gamma=0.99):
    """Extract the policy given a value function
        v: Value policy
    Returns:
        policy based on algorithm
    """

    env_nS = env.nrow * env.ncol        # total number of possible states = 64

    policy = np.zeros((env.nrow, env.ncol))
    
    for s in range(env_nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next_state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))

        policy[s] = np.argmax(q_sa)
    return policy

def value_iteration(env, gamma=0.99):
    env_nS = env.nrow * env.ncol        # total number of possible states = 64
    env_nA = 4                          # total number of possible actions = 4

    v = np.zeros(env_nS)        # initialize value-function

    # max_iterations = 100000
    max_iterations = 100
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env_nS):
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env_nA)]        # env.P[s][a] Probability of reaching successor state s and reward r
            v[s] = max(q_sa)

        if (np.sum(np.fabs(prev_v)) <= eps):        # np.fabs -> returns absolute values of the data in x.
            print('Value-iteration converged at iteration # %d.' %(i+1))
            break

    return v

if __name__ == '__main__':
    env_name = 'FrozenLake8x8-v1'
    gamma = 1.0
    env = gym.make(env_name, is_slippery=False)

    optimal_v = value_iteration(env, gamma)
    policy = extract_policy(env, optimal_v, gamma)
    policy_score = evaluate_policy(env, policy, gamma, n=1000)
    print('Policy average score = ', policy_score)