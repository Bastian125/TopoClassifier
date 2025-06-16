"""
All models used in train.py are defined here.
"""

# ---------- Imports ---------- #
import torch.nn as nn

# ---------- Models ---------- #

class DNNModel(nn.Module):
    """
    Deep Neural Network model for binary classification.
    Final layer does NOT include sigmoid; use BCEWithLogitsLoss instead.
    """

    def __init__(self, input_dim):
        super(DNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.model(x)