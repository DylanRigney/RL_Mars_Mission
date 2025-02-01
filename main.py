from env.mars_env import MarsEnv

def main():
    env = MarsEnv()
    env.reset()

    for _ in range(1000):
        action = env.sample_action()
        obs, reward, done, info = env.step(action)
        env.render()
        
        if done:
            env.reset()

if __name__ == "__main__":
    main()