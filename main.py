import random
import time
import pybullet as p
import numpy as np

from env.mars_env import MarsEnv

def main():
    env = MarsEnv()
    done = False
    total_reward = 0

    while not done:
        action = (random.randint(-10, 10), random.randint(-10, 10)) # TODO: Implement sample_action
        obs, reward, done = env.step(action)
        
        total_reward += reward

        print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Total Reward: {total_reward}")
        

    p.disconnect()
    print("Episode finished with total reward:", total_reward)

if __name__ == "__main__":
    main()