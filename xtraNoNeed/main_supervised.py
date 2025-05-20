# main_supervised.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import trange
from collections import Counter

from models.transformer_encoder import TransformerEncoder
from models.dqn_agent import DQNAgent
from utils.data_loader import create_dataset
from utils.metrics import compute_metrics
from config import LEARNING_RATE, BATCH_SIZE, NUM_CLASSES

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# Load and Prepare Data
# ---------------------
X_static, X_time, y = create_dataset()
print("Loaded dataset shapes:")
print("Static:", X_static.shape, "Time-Series:", X_time.shape, "Labels:", y.shape)

X_static = np.array(X_static)
X_time = np.array(X_time)
y = np.array(y)

counter = Counter(y)
print("Label distribution:", counter)
total_samples = len(y)
class_weights = {}
for label in counter:
    class_weights[label] = total_samples / (NUM_CLASSES * counter[label])
print("Class weights:", class_weights)

# ---------------------
# Train-Test Split
# ---------------------
X_static_train, X_static_test, X_time_train, X_time_test, y_train, y_test = train_test_split(
    X_static, X_time, y, test_size=0.2, random_state=42, stratify=y)
print("Train/Test sizes:", len(y_train), len(y_test))

static_train = torch.tensor(X_static_train, dtype=torch.float32).to(device)
time_train = torch.tensor(X_time_train, dtype=torch.float32).to(device)
labels_train = torch.tensor(y_train, dtype=torch.long).to(device)

static_test = torch.tensor(X_static_test, dtype=torch.float32).to(device)
time_test = torch.tensor(X_time_test, dtype=torch.float32).to(device)
labels_test = torch.tensor(y_test, dtype=torch.long).to(device)

num_train_samples = static_train.size(0)

# ---------------------
# Initialize Model
# ---------------------
# We'll use the same architecture (Transformer + DQN agent) but train it in a supervised way.
transformer = TransformerEncoder().to(device)
static_input_dim = static_train.shape[1]
dqn_agent = DQNAgent(static_input_dim=static_input_dim).to(device)

# ---------------------
# Optimizer and Loss Function
# ---------------------
# Use a lower learning rate for stability
optimizer = optim.Adam(list(transformer.parameters()) + list(dqn_agent.parameters()), lr=1e-4, weight_decay=1e-5)
weight_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)

# ---------------------
# Supervised Training Loop
# ---------------------
num_epochs = 50  # You can adjust the number of epochs
for epoch in trange(num_epochs):
    permutation = torch.randperm(num_train_samples)
    epoch_loss = 0.0
    for i in range(0, num_train_samples, BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_static = static_train[indices]
        batch_time = time_train[indices]
        batch_labels = labels_train[indices]

        transformer_out = transformer(batch_time)
        q_values = dqn_agent(transformer_out, batch_static)
        # Optionally clamp Q-values to avoid extreme values
        q_values = torch.clamp(q_values, -100, 100)
        
        loss = criterion(q_values, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(transformer.parameters()) + list(dqn_agent.parameters()), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# ---------------------
# Evaluation
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
