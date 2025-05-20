import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TRANSFORMER_EMBED_DIM, NUM_CLASSES

class DQNAgent(nn.Module):
    def __init__(self, static_input_dim, dropout=0.3):
        super(DQNAgent, self).__init__()
        self.fc_static = nn.Sequential(
            nn.Linear(static_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(TRANSFORMER_EMBED_DIM + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.out = nn.Linear(32, NUM_CLASSES)
    
    def forward(self, transformer_state, static_features):
        static_out = self.fc_static(static_features)
        combined = torch.cat((transformer_state, static_out), dim=1)
        x = self.fc1(combined)
        x = self.fc2(x)
        q_values = self.out(x)
        return q_values

if __name__ == '__main__':
    dummy_transformer_state = torch.rand(8, TRANSFORMER_EMBED_DIM)
    dummy_static = torch.rand(8, 4)  # assuming 4 static features
    model = DQNAgent(static_input_dim=4)
    q_vals = model(dummy_transformer_state, dummy_static)
    print("Q-values shape:", q_vals.shape)  # Expected: (8, NUM_CLASSES)
