# https://towardsdatascience.com/hands-on-reinforcement-learning-course-part-1-269b50e39d08
"""
Boiler plate code for making an agent learn via
"""

import random

def train(n_episodes: int):
    """Pseudo-code  of  a Re-inforcement Learning agent training loop

    Args:
        n_episodes (int): No. of episdoes the agent will run to get optimal policy.
    """

    # get the environment you are trying to learn.
    env = load_env()

    # get the agent: Python object that wraps all agent policy (or value function) parameters. and action generation methods.
    agent = get_rl_agent()

    for episode in range(0, n_episodes):

        # random start of the event.
        state = env.reset()

        # epsilon is parameter that controls the exploitation-exploration tradeoff. Probability that agent chooses a random function or the best function.
        # It is a good practice to set a decaying value for Epsilon/eta so that agent prefers to learn quickly.
        epsilon = get_epsilon()         # also called eta.

        done = False

        while not done:

            if random.uniform(0, 1) < epsilon:
                # explore the action space.
                action = env.action_space.sample()
            else:
                # exploit learned value (or policy)
                action = agent.get_best_action(state)

            
            # environment transitions to next state and maybe rewards the agent.
            next_state, reward, done, info = env.step(action)

            agent.update_parameters(state, action, reward, next_state)

            state = next_state


