# main_cv.py
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from sklearn.model_selection import KFold
from collections import Counter
from tqdm import trange

from models.transformer_encoder import TransformerEncoder
from models.dqn_agent import DQNAgent
from utils.data_loader import create_dataset
from utils.metrics import compute_metrics
from config import (
    LEARNING_RATE, BATCH_SIZE, NUM_CLASSES, TARGET_UPDATE_FREQ, 
    NUM_EPISODES, EPSILON_START, EPSILON_END, EPSILON_DECAY, SEQUENCE_LENGTH
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
X_static, X_time, y = create_dataset()
print("Static Features shape:", X_static.shape)
print("Time-Series Features shape:", X_time.shape)
print("Labels shape:", y.shape)

# Convert static features to numpy (if X_static is a DataFrame)
X_static_np = X_static.to_numpy() if hasattr(X_static, 'to_numpy') else np.array(X_static)
y_np = y

# Set hybrid loss balance: lower RL weight so that the supervised loss has greater influence.
alpha = 0.1  # 10% RL loss, 90% supervised loss

# Adjusted class weights: lower control weight (label 0) to 1.5, depressed remains 1.0.
adjusted_weights = torch.tensor([1.5, 1.0], dtype=torch.float32).to(device)

# Define KFold cross-validation (5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
fold_metrics = []

for train_index, val_index in kf.split(X_static_np):
    print(f"\n--- Fold {fold} ---")
    # Create train and validation splits
    X_static_train, X_static_val = X_static_np[train_index], X_static_np[val_index]
    X_time_train, X_time_val = X_time[train_index], X_time[val_index]
    y_train, y_val = y_np[train_index], y_np[val_index]
    
    # Convert to torch tensors
    static_train = torch.tensor(X_static_train, dtype=torch.float32).to(device)
    time_train = torch.tensor(X_time_train, dtype=torch.float32).to(device)
    labels_train = torch.tensor(y_train, dtype=torch.long).to(device)
    
    static_val = torch.tensor(X_static_val, dtype=torch.float32).to(device)
    time_val = torch.tensor(X_time_val, dtype=torch.float32).to(device)
    labels_val = torch.tensor(y_val, dtype=torch.long).to(device)
    
    num_train_samples = static_train.size(0)
    
    # Initialize models for this fold
    transformer = TransformerEncoder(dropout=0.3).to(device)
    dqn_agent = DQNAgent(static_input_dim=static_train.shape[1], dropout=0.3).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(list(transformer.parameters()) + list(dqn_agent.parameters()), lr=LEARNING_RATE)
    
    # Use a simple replay buffer (list) for one-step episodes
    replay_buffer = []
    
    epsilon = EPSILON_START
    
    # Training loop (RL + supervised hybrid loss)
    for episode in trange(NUM_EPISODES, desc=f"Training Fold {fold}"):
        idx = random.randint(0, num_train_samples - 1)
        state_static = static_train[idx].unsqueeze(0)   # (1, static_dim)
        state_time = time_train[idx].unsqueeze(0)         # (1, SEQUENCE_LENGTH)
        true_label = int(labels_train[idx].item())
        
        # Forward pass: get state representation and Q-values
        transformer_state = transformer(state_time)
        q_values = dqn_agent(transformer_state, state_static)
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, NUM_CLASSES - 1)
        else:
            action = torch.argmax(q_values, dim=1).item()
        
        # Define reward: +1 if correct, -1 otherwise
        reward = 1.0 if action == true_label else -1.0
        
        # Store experience (one-step episode)
        replay_buffer.append((state_static, state_time, action, reward, true_label))
        if len(replay_buffer) > BATCH_SIZE:
            replay_buffer.pop(0)
        
        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            batch_static = torch.cat([x[0] for x in batch], dim=0)
            batch_time = torch.cat([x[1] for x in batch], dim=0)
            batch_actions = torch.tensor([x[2] for x in batch], dtype=torch.long).to(device)
            batch_rewards = torch.tensor([x[3] for x in batch], dtype=torch.float32).to(device)
            batch_labels = torch.tensor([x[4] for x in batch], dtype=torch.long).to(device)
            
            transformer_out = transformer(batch_time)
            current_q_values = dqn_agent(transformer_out, batch_static)
            
            # Compute RL loss: SmoothL1Loss between Q-value for the chosen action and reward
            chosen_q = current_q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            rl_loss = nn.SmoothL1Loss()(chosen_q, batch_rewards)
            
            # Compute supervised loss: CrossEntropyLoss with adjusted class weights
            sup_loss = nn.CrossEntropyLoss(weight=adjusted_weights)(current_q_values, batch_labels)
            
            total_loss = alpha * rl_loss + (1 - alpha) * sup_loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(transformer.parameters()) + list(dqn_agent.parameters()), max_norm=1.0)
            optimizer.step()
        
        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    # ---------------
    # Threshold Calibration on Validation Set
    # ---------------
    with torch.no_grad():
        transformer_out_val = transformer(time_val)
        q_values_val = dqn_agent(transformer_out_val, static_val)
        probs_val = F.softmax(q_values_val, dim=1).cpu().numpy()
        # Probability for the "depressed" class is assumed at index 1
        probs_depr = probs_val[:, 1]
        true_labels_val = labels_val.cpu().numpy()
        
        best_thresh = None
        best_bal_acc = -np.inf
        # Set target specificity range to [0.75, 0.85] to allow increased sensitivity
        target_spe_min = 0.70
        target_spe_max = 0.80
        
        for thresh in np.linspace(0.1, 0.9, 81):
            preds = (probs_depr >= thresh).astype(int)
            metrics = compute_metrics(true_labels_val, preds)
            bal_acc = 0.5 * (metrics['SEN'] + metrics['SPE'])
            if target_spe_min <= metrics['SPE'] <= target_spe_max and bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_thresh = thresh
        
        # Fallback: if no threshold meets the target, choose the one with maximum balanced accuracy
        if best_thresh is None:
            for thresh in np.linspace(0.1, 0.9, 81):
                preds = (probs_depr >= thresh).astype(int)
                metrics = compute_metrics(true_labels_val, preds)
                bal_acc = 0.5 * (metrics['SEN'] + metrics['SPE'])
                if bal_acc > best_bal_acc:
                    best_bal_acc = bal_acc
                    best_thresh = thresh
        
        print(f"Fold {fold} optimal threshold: {best_thresh:.2f} (Balanced Accuracy: {best_bal_acc:.2f})")
        preds_val = (probs_depr >= best_thresh).astype(int)
        metrics_val = compute_metrics(true_labels_val, preds_val)
        print(f"Fold {fold} Metrics: {metrics_val}")
        fold_metrics.append(metrics_val)
    
    fold += 1

# Average metrics across folds
avg_acc = np.mean([m['ACC'] for m in fold_metrics])
avg_sen = np.mean([m['SEN'] for m in fold_metrics])
avg_spe = np.mean([m['SPE'] for m in fold_metrics])
avg_f1 = np.mean([m['F1'] for m in fold_metrics])

print("\nAverage Cross-Validation Metrics:")
print(f"ACC: {avg_acc:.2f}, SEN: {avg_sen:.2f}, SPE: {avg_spe:.2f}, F1: {avg_f1:.2f}")
