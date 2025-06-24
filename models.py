"""
All models used in train.py are defined here.
"""

# ---------- Imports ---------- #
import torch
import torch.nn as nn
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


class GATNet(nn.Module):
    def __init__(self, in_channels):
        super(GATNet, self).__init__()

        self.conv1 = GATConv(in_channels, 16, heads=8, dropout=0.1)
        self.conv2 = GATConv(16 * 8, 32, heads=8, dropout=0.1)

        self.Linear_node1 = torch.nn.Linear(16, 32)
        self.Linear_node2 = torch.nn.Linear(32, 32)

        self.Linear_node_After1 = torch.nn.Linear(128, 64)
        self.Linear_node_After2 = torch.nn.Linear(64, 64)

        self.Linear_node_After3 = torch.nn.Linear(256, 128)
        self.Linear_node_After4 = torch.nn.Linear(128, 64)

        self.Linear_final1 = torch.nn.Linear(160, 128)
        self.Linear_final2 = torch.nn.Linear(128, 64)
        self.Linear_final3 = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index):

        gg1 = self.Linear_node1(x)
        gg1 = torch.relu(gg1)

        gg2 = self.Linear_node2(gg1)
        gg2 = torch.relu(gg2)

        x1 = self.conv1(x, edge_index)
        x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = torch.relu(x2)

        x_after1 = self.Linear_node_After1(x1)
        x_after1 = torch.relu(x_after1)
        x_after1 = self.Linear_node_After2(x_after1)
        x_after1 = torch.relu(x_after1)

        x_after2 = self.Linear_node_After3(x2)
        x_after2 = torch.relu(x_after2)
        x_after2 = self.Linear_node_After4(x_after2)

        x_after2 = torch.relu(x_after2)

        xfinal = torch.cat((gg2, x_after1, x_after2), dim=1)
        xfinal = self.Linear_final1(xfinal)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final2(xfinal)
        xfinal = torch.relu(xfinal)
        xfinal = self.Linear_final3(xfinal)

        return xfinal
