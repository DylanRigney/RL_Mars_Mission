import time
import pybullet as p
import numpy as np
import torch
from env.mars_env import MarsEnv  # Make sure the path matches your project structure
from agents.dqn_agent import DQNAgent

def main():
    # Initialize the Mars environment
    env = MarsEnv()
    
    # Retrieve observation and action space dimensions from the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create the DQN agent with the appropriate dimensions
    agent = DQNAgent(state_size, action_size)
    
    # Set the number of episodes for training
    EPISODES = 1000
    TARGET_UPDATE_FREQ = 10  # How often to update the target network
    
    for episode in range(EPISODES):
        state = env.reset()  # Reset the environment at the start of each episode
        total_reward = 0
        done = False
        
        while not done:
            # Agent selects an action using epsilon-greedy strategy
            action = agent.act(state)
            # Environment takes the action and returns next_state, reward, done flag, and extra info
            next_state, reward, done, _ = env.step(action)
            
            # Store the experience in the replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Train the agent using a minibatch from the replay buffer
            agent.replay()
            
            state = next_state
            total_reward += reward
        
        # Periodically update the target network for stability
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        print(f"Episode {episode + 1}/{EPISODES}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
        time.sleep(0.1)  # Optional: slow down logging for readability

    # Save the trained model weights for later use
    torch.save(agent.q_network.state_dict(), "models/dqn_mars_rover.pth")
    print("Training complete! Model saved.")

    p.disconnect()

if __name__ == "__main__":
    main()