import time
import torch
import numpy as np
from env.mars_env import MarsEnv  # Ensure this path matches your project structure
from agents.dqn_agent import DQNAgent  # Ensure this path matches your project structure

def main():
    # Set training hyperparameters
    EPISODES = 100          # Total number of episodes for training
    TARGET_UPDATE_FREQ = 10   # Frequency (in episodes) to update the target network
    RENDER_EVERY = 50         # Render environment every X episodes (optional)
    
    # Initialize the Mars environment
    env = MarsEnv()
    
    # Get state and action dimensions from the environment's spaces
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize the DQN agent with the given state and action dimensions
    agent = DQNAgent(state_size, action_size)
    
    # Training loop: iterate over episodes
    for episode in range(EPISODES):
        state = env.reset()  # Reset the environment and obtain the initial state
        total_reward = 0
        done = False
        
        # Loop for each step of the episode until termination
        step_count = 0
        while not done:
            # Agent selects an action using epsilon-greedy policy
            action = agent.act(state)
            
            # Environment processes the action and returns next_state, reward, done flag, and extra info
            step_count += 1
            next_state, reward, done, _ = env.step(action, step_count)
            
            # Store the experience in the replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Train the agent with a minibatch from the replay buffer
            agent.replay()
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
        
        # Update the target network every TARGET_UPDATE_FREQ episodes for stable learning
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        # Print training progress for this episode
        print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
        
        # Optionally render the environment every few episodes (if render is implemented)
        if episode % RENDER_EVERY == 0:
            # You can implement a render() method in your MarsEnv if desired.
            env.render()
            time.sleep(1)  # Pause briefly for visualization
    
    # Save the trained Q-network weights for later evaluation or deployment
    torch.save(agent.q_network.state_dict(), "models/dqn_mars_rover.pth")
    print("Training complete! Model saved.")
    
    # Close the environment connection
    env.close()

if __name__ == "__main__":
    main()
