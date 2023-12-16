import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
import logging
from tqdm.auto import tqdm
from transformers import FuyuForCausalLM, FuyuProcessor
from peft import LoraConfig
from torchinfo import summary
import os
import random
from collections import deque

# Set the root logger level to ERROR
logging.basicConfig(level=logging.ERROR)

# Iterate over all existing loggers and set their levels to ERROR
for logger in logging.root.manager.loggerDict.values():
    if isinstance(logger, logging.Logger):  # Check if it is a Logger instance
        logger.setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else None)

# Create the FrozenLake environment
# env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
desc=["SF", "HG"]
# env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False, render_mode="human")
env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False)

# Define the custom Fuyu model for the task
class CustomFuyu(nn.Module):
    """
    Custom Fuyu model for the FrozenLake environment.
    """
    def __init__(self, action_space_size):
        super().__init__()
        self.fuyu = FuyuForCausalLM.from_pretrained("adept/fuyu-8b",
                                                    load_in_4bit=True,
                                                    output_hidden_states=True,
                                                    bnb_4bit_use_double_quant=True,
                                                    bnb_4bit_quant_type="nf4",
                                                    bnb_4bit_compute_dtype=torch.bfloat16)
        
        lora_config = LoraConfig(target_modules=["query_key_value"], init_lora_weights=False)
        self.fuyu.add_adapter(lora_config, adapter_name="lora")
        self.fuyu.language_model.lm_head = nn.Linear(self.fuyu.config.hidden_size, action_space_size).half()

    def forward(self, input_ids, attention_mask=None):
        return self.fuyu(input_ids, attention_mask=attention_mask).logits[:, -1, :]

# Initialize the model and optimizer
model = CustomFuyu(env.action_space.n).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")

MAX_SEQ_LEN = 20

# DQN-specific hyperparameters
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
replay_buffer = deque(maxlen=10000)

# Function to get state in the required format
def process_state(state):
    state_text = f"The current state is {state}."
    # print(state_text)
    inputs = processor(text=state_text, return_tensors="pt").to(device)
    return inputs

# Function to choose an action based on epsilon-greedy policy
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        with torch.no_grad():
            state_input = process_state(state)
            q_values = model(**state_input)
        return torch.argmax(q_values).item()  # Exploit

# Function to perform one step in the environment
def step_env(state, action):
    next_state, reward, done, _, _ = env.step(action)
    return next_state, reward, done

# Function to save experience in replay buffer
def store_experience(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

# Function to sample a batch of experiences from the buffer
def sample_batch(batch_size):
    return random.sample(replay_buffer, batch_size)

# Function to train the model
def train_model(batch_size):
    if len(replay_buffer) < batch_size:
        return  # Not enough samples

    minibatch = sample_batch(batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    # Convert to tensors
    states = torch.stack([process_state(s)['input_ids'].squeeze(0) for s in states])
    next_states = torch.stack([process_state(s)['input_ids'].squeeze(0) for s in next_states])
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(1).half()
    dones = torch.tensor(dones, dtype=torch.float32).to(device).unsqueeze(1).half()

    # Get Q values for current states
    q_values = model(states).gather(1, actions)

    # Get max Q values for next states
    with torch.no_grad():
        max_next_q_values = model(next_states).max(1)[0].unsqueeze(1)
    target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

    # Compute loss and update model
    loss = F.mse_loss(q_values, target_q_values)  # Ensure both tensors have shape [batch_size, 1]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Training loop
num_episodes = 1000
ten_episode_win_count = 0
for episode in tqdm(range(num_episodes), desc="Training Progress"):
    state = env.reset()
    done = False
    while not done:
        if isinstance(state, tuple):
            state = state[0]
        action = choose_action(state, epsilon)
        next_state, reward, done = step_env(state, action)
        store_experience(state, action, reward, next_state, done)
        state = next_state
        ten_episode_win_count += reward

        train_model(batch_size)

    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 10 == 0:
        print(f"Episode: {episode}, 10-Episode Win Rate: {ten_episode_win_count}, Epsilon: {epsilon}")
        ten_episode_win_count = 0

env.close()
