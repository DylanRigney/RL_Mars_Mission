import torch
import time
from agents.dqn_agent import DQNAgent
from env.mars_env import MarsEnv

def train_dqn():
    # hyperparameters
    EPISODES = 1000
    TARGET_UPDATE_FREQ = 10  # Number of episodes between target network updates
    RENDER_EVERY = 50 # Render the environment every X episodes

    # Initialize environment and agent
    env = MarsEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Choose an action using epsilon-greedy policy
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            # Store experience in the replay buffer
            agent.remember(state, action, reward, next_state, done)

            # Train the agent
            agent.replay()

            # Update state
            state = next_state
            total_reward += reward

        # Update target network periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Print episode statistics
        print(f"Episode {episode + 1}/{EPISODES}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

        # Render every few episodes for debugging
        if episode % RENDER_EVERY == 0:
            # If I implemnt a render method
         # env.render()
            time.sleep(1)

    # Save the trained model
    torch.save(agent.q_network.state_dict(), "dqn_mars_rover.pth")
    print("Training complete! Model saved.")
    env.close()

if __name__ == "__main__":
    train_dqn()