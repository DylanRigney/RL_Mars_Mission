import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Q-Network Architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 64)    # First hidden layer
        self.fc2 = nn.Linear(64, 64)            # Second hidden layer
        self.output = nn.Linear(64, action_size) # Output layer (one Q-value per action)

    def forward(self, state):
        x = torch.relu(self.fc1(state))  # Apply ReLU activation
        x = torch.relu(self.fc2(x))
        return self.output(x)            # Raw Q-values (no activation function)
        

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma                  # Discount factor
        self.epsilon = epsilon              # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.epsilon_min = epsilon_min      # Minimum value of epsilon
        self.batch_size = batch_size        # Size of training batch
        self.memory = deque(maxlen=memory_size)  # Experience replay buffer

        # Initialize the Q-Network and Target Network
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Copy weights to target network
        self.target_network.eval()  # Target network is not trained directly

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()  # Mean Squared Error loss for Q-value predictions

    def remember(self, state, action, reward, next_state, done):
        # Store experience in the replay buffer for experience replay
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size)) # Explore: random action
        state = torch.FloatTensor(state).unsqueeze(0) # Convert state to tensor
        q_values = self.q_network(state)                   # Predict Q-values
        return torch.argmax(q_values).item()                # Exploit: choose action with highest Q-value

    def replay(self):
        if len(self.memory) < self.batch_size:
                return  # Not enough experience to train on

        # Sample a batch of experiences from the replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert states, actions, rewards, next_states, and dones to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions)
            
        # Target Q-values using the target network
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the Q-Network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        # Periodically update the target network to match the Q-network
        self.target_network.load_state_dict(self.q_network.state_dict())