import gym
import time


print("This is a test change!")
print()

env = gym.make("LunarLander-v2", render_mode='human')
# env = gym.make("BipedalWalker-v3", render_mode='human')
env.action_space.seed(42)

# comment this out if doesn't work
observation, info = env.reset(seed=42, return_info=True)

t = 0
for _ in range(1000):
    # print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    # time.sleep(0.1)

    if done:
        # comment out if code doesn't work
        observation, info = env.reset(return_info=True)
        # break
        # repeat = input("Do you want to RUN the Simulation\'s again? (Y/n) \n")

        # if repeat == 'N' or repeat == 'n':
        #     break

env.close()