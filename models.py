"""
All models used in train.py are defined here.
"""

# ---------- Imports ---------- #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


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
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.model(x)


class GAT(nn.Module):
    def __init__(self, in_channels):
        super(GAT, self).__init__()

        # GAT layers
        self.conv1 = GATConv(in_channels, 16, heads=8, dropout=0.1)
        self.conv2 = GATConv(16 * 8, 32, heads=8, dropout=0.1)

        # Feed-forward pathway for raw input features
        self.Linear_node1 = nn.Linear(in_channels, 32)
        self.Linear_node2 = nn.Linear(32, 32)

        # MLP on GAT conv1 output
        self.Linear_node_After1 = nn.Linear(128, 64)
        self.Linear_node_After2 = nn.Linear(64, 64)

        # MLP on GAT conv2 output
        self.Linear_node_After3 = nn.Linear(256, 128)
        self.Linear_node_After4 = nn.Linear(128, 64)

        # Final classifier head
        self.Linear_final1 = nn.Linear(160, 128)
        self.Linear_final2 = nn.Linear(128, 64)
        self.Linear_final3 = nn.Linear(64, 1)

    def forward(self, x, edge_index):
        # Process raw input features
        gg1 = F.relu(self.Linear_node1(x))
        gg2 = F.relu(self.Linear_node2(gg1))

        # GAT path
        x1 = F.relu(self.conv1(x, edge_index))  # [N, 128]
        x2 = F.relu(self.conv2(x1, edge_index))  # [N, 256]

        # Feed-forward on x1 and x2
        x_after1 = F.relu(self.Linear_node_After1(x1))
        x_after1 = F.relu(self.Linear_node_After2(x_after1))

        x_after2 = F.relu(self.Linear_node_After3(x2))
        x_after2 = F.relu(self.Linear_node_After4(x_after2))

        # Concatenate all processed representations
        xfinal = torch.cat([gg2, x_after1, x_after2], dim=1)  # [N, 160]
        xfinal = F.relu(self.Linear_final1(xfinal))
        xfinal = F.relu(self.Linear_final2(xfinal))
        xfinal = self.Linear_final3(xfinal)

        return xfinal
