# main.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tqdm import trange
from collections import Counter

from models.transformer_encoder import TransformerEncoder
from models.dqn_agent import DQNAgent
from utils.data_loader import create_dataset
from utils.metrics import compute_metrics
from config import (LEARNING_RATE, BATCH_SIZE, NUM_CLASSES, TARGET_UPDATE_FREQ, GAMMA, 
                    NUM_EPISODES, EPSILON_START, EPSILON_END, EPSILON_DECAY)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# Experience Replay Buffer
# ---------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def push(self, state_static, state_time, action, reward, next_state_static, next_state_time, done):
        self.buffer.append((state_static, state_time, action, reward, next_state_static, next_state_time, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# ---------------------
# Load and Prepare Data
# ---------------------
X_static, X_time, y = create_dataset()
print("Loaded dataset shapes:")
print("Static:", X_static.shape, "Time-Series:", X_time.shape, "Labels:", y.shape)

# Convert to numpy arrays
X_static = np.array(X_static)
X_time = np.array(X_time)
y = np.array(y)

# Compute class counts and weights
counter = Counter(y)
print("Label distribution:", counter)
total_samples = len(y)
class_weights = {}
for label in counter:
    # Weight = total_samples / (NUM_CLASSES * count)
    class_weights[label] = total_samples / (NUM_CLASSES * counter[label])
print("Class weights:", class_weights)

# ---------------------
# Train-Test Split (80-20 with stratification)
# ---------------------
X_static_train, X_static_test, X_time_train, X_time_test, y_train, y_test = train_test_split(
    X_static, X_time, y, test_size=0.2, random_state=42, stratify=y)
print("Train/Test sizes:", len(y_train), len(y_test))

# Convert to torch tensors
static_train = torch.tensor(X_static_train, dtype=torch.float32).to(device)
time_train = torch.tensor(X_time_train, dtype=torch.float32).to(device)
labels_train = torch.tensor(y_train, dtype=torch.long).to(device)

static_test = torch.tensor(X_static_test, dtype=torch.float32).to(device)
time_test = torch.tensor(X_time_test, dtype=torch.float32).to(device)
labels_test = torch.tensor(y_test, dtype=torch.long).to(device)

num_train_samples = static_train.size(0)

# ---------------------
# Initialize Models
# ---------------------
transformer = TransformerEncoder().to(device)
static_input_dim = static_train.shape[1]  # e.g., 8 features
dqn_agent = DQNAgent(static_input_dim=static_input_dim).to(device)
# Target network is a copy of the DQN agent
target_agent = DQNAgent(static_input_dim=static_input_dim).to(device)
target_agent.load_state_dict(dqn_agent.state_dict())
target_agent.eval()  # target network in eval mode

# ---------------------
# Optimizers and Loss Functions
# ---------------------
# For the RL phase, use a lower learning rate with weight decay and SmoothL1Loss for stability.
optimizer_rl = optim.Adam(
    list(transformer.parameters()) + list(dqn_agent.parameters()), 
    lr=LEARNING_RATE / 10, 
    weight_decay=1e-5
)
# Using SmoothL1Loss (Huber Loss) instead of MSELoss
criterion_rl = nn.SmoothL1Loss()

# For supervised fine-tuning, use weighted cross-entropy loss.
weight_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)
criterion_sup = nn.CrossEntropyLoss(weight=weight_tensor)

# ---------------------
# RL Hyperparameters and Replay Buffer
# ---------------------
buffer_capacity = 10000
replay_buffer = ReplayBuffer(buffer_capacity)
epsilon = EPSILON_START

# ---------------------
# Phase 1: RL Training Loop (One-step Episodes)
# ---------------------
print("\nStarting RL Training...\n")
for episode in trange(NUM_EPISODES):
    idx = random.randint(0, num_train_samples - 1)
    state_static = static_train[idx].unsqueeze(0)  # (1, feature_dim)
    state_time = time_train[idx].unsqueeze(0)        # (1, SEQUENCE_LENGTH)
    true_label = int(labels_train[idx].item())

    transformer_state = transformer(state_time)  # (1, embed_dim)
    
    # Epsilon-greedy action selection
    if random.random() < epsilon:
        action = random.randint(0, NUM_CLASSES - 1)
    else:
        with torch.no_grad():
            q_values = dqn_agent(transformer_state, state_static)
            # Clamp Q-values to avoid extreme values
            q_values = torch.clamp(q_values, -100, 100)
            action = torch.argmax(q_values, dim=1).item()

    # Reward: weighted reward (+weight if correct, -weight if incorrect)
    weight_val = class_weights[true_label]
    reward = weight_val if action == true_label else -weight_val
    done = True  # one-step episode

    next_state_static = torch.zeros_like(state_static)
    next_state_time = torch.zeros_like(state_time)

    replay_buffer.push(state_static, state_time, action, reward, next_state_static, next_state_time, done)

    if len(replay_buffer) >= BATCH_SIZE:
        batch = replay_buffer.sample(BATCH_SIZE)
        batch_state_static = torch.cat([item[0] for item in batch], dim=0)
        batch_state_time = torch.cat([item[1] for item in batch], dim=0)
        batch_action = torch.tensor([item[2] for item in batch], dtype=torch.long).to(device)
        batch_reward = torch.tensor([item[3] for item in batch], dtype=torch.float32).to(device)

        transformer_out = transformer(batch_state_time)
        current_q_values = dqn_agent(transformer_out, batch_state_static)
        # Clamp current Q-values
        current_q_values = torch.clamp(current_q_values, -100, 100)
        current_q = current_q_values.gather(1, batch_action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            target_q = batch_reward

        loss_rl = criterion_rl(current_q, target_q)
        optimizer_rl.zero_grad()
        loss_rl.backward()
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(list(transformer.parameters()) + list(dqn_agent.parameters()), max_norm=1.0)
        optimizer_rl.step()

    if episode % TARGET_UPDATE_FREQ == 0:
        target_agent.load_state_dict(dqn_agent.state_dict())

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

# ---------------------
# Phase 2: Supervised Fine-Tuning
# ---------------------
print("\nStarting Supervised Fine-Tuning...\n")
num_finetune_epochs = 20
optimizer_sup = optim.Adam(
    list(transformer.parameters()) + list(dqn_agent.parameters()), 
    lr=1e-4,
    weight_decay=1e-5
)

for epoch in trange(num_finetune_epochs):
    permutation = torch.randperm(num_train_samples)
    epoch_loss = 0.0
    for i in range(0, num_train_samples, BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_static = static_train[indices]
        batch_time = time_train[indices]
        batch_labels = labels_train[indices]

        transformer_out = transformer(batch_time)
        q_values = dqn_agent(transformer_out, batch_static)
        # Clamp Q-values to a safe range before computing loss
        q_values = torch.clamp(q_values, -100, 100)
        
        loss_sup = criterion_sup(q_values, batch_labels)
        optimizer_sup.zero_grad()
        loss_sup.backward()
        torch.nn.utils.clip_grad_norm_(list(transformer.parameters()) + list(dqn_agent.parameters()), max_norm=1.0)
        optimizer_sup.step()
        epoch_loss += loss_sup.item()
    print(f"Finetune Epoch {epoch+1}/{num_finetune_epochs}, Loss: {epoch_loss:.4f}")

# ---------------------
# Evaluation (Supervised Forward Pass)
# ---------------------
with torch.no_grad():
    transformer_out_test = transformer(time_test)
    q_values_test = dqn_agent(transformer_out_test, static_test)
    q_values_test = torch.clamp(q_values_test, -100, 100)
    predictions = torch.argmax(q_values_test, dim=1).cpu().numpy()
    true_labels = labels_test.cpu().numpy()
    
metrics_result = compute_metrics(true_labels, predictions)
print("\nEvaluation Metrics:")
print(metrics_result)
