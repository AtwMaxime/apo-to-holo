import os
import pandas as pd
import torch
from torch_geometric.data import Data

# Folder containing your CSV files
graph_folder = r"apo"

# Initialize a list to store the Data objects
graph_data_list = []

# Step 1: Iterate through all the CSV files in the folder
for filename in os.listdir(graph_folder):
    if filename.endswith("_links.csv"):
        # Get the corresponding node file
        node_filename = filename.replace("_links.csv", "_nodes.csv")
        
        # Full paths to the edge and node files
        link_file_path = os.path.join(graph_folder, filename)
        node_file_path = os.path.join(graph_folder, node_filename)
        
        # Step 2: Load the edge data (links)
        edge_df = pd.read_csv(link_file_path)
        
        # Extract the atom indices for the source and target nodes
        edge_index = torch.tensor(
            edge_df[['atom_index1', 'atom_index2']].values.T, dtype=torch.long
        )
        
        # Extract relevant edge attributes
        edge_attr = edge_df[['distance', 'area', 'boundary', 'voromqa_energy', 
                             'seq_sep_class', 'covalent_bond', 'hbond']].values
        
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Step 3: Load the node data (node features)
        node_df = pd.read_csv(node_file_path)
        
        # Select relevant node features for the graph (e.g., coordinates, radius, SAS potential)
        node_features = node_df[['center_x', 'center_y', 'center_z', 'radius', 
                                 'voromqa_sas_potential', 'residue_mean_sas_potential', 
                                 'residue_sum_sas_potential', 'residue_size']].values
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        ground_truth = node_df['ground_truth'].values  # Extract ground truth
        
        # Step 4: Create a Data object
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        
        data.y = torch.tensor(ground_truth, dtype=torch.float)
        
        # Step 5: Store the Data object in the list
        graph_data_list.append(data)

# Now, you have all the graphs in the list `graph_data_list` as PyTorch Geometric Data objects
print(f"Loaded {len(graph_data_list)} graphs.")

#%%
import matplotlib.pyplot as plt
import networkx as nx
import torch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Let's assume you have the following from your Data object:
data = graph_data_list[3]  # Get the first graph data
edge_index = data.edge_index.numpy()  # Convert edge_index to numpy array
node_features = data.x.numpy()  # Convert node features to numpy array
node_labels = data.y.numpy()  # Ground truth labels (already in [0, 1])

# Extract the 3D coordinates (assuming they are in the first three columns)
coords = node_features[:, :3]  # Get center_x, center_y, center_z

# Step 1: Create a NetworkX graph from the edge_index
G = nx.Graph()  # Create an undirected graph; change to nx.DiGraph() for directed graphs

# Add edges from edge_index
edges = [(int(edge_index[0][i]), int(edge_index[1][i])) for i in range(edge_index.shape[1])]
G.add_edges_from(edges)

# Step 2: Prepare for 3D plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Step 3: Define a custom colormap from blue to red
blue_red_cmap = LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])

# Step 4: Use the custom colormap to map values directly to colors
node_colors = blue_red_cmap(node_labels)  # Map normalized values (0 to 1) to colors

# Step 5: Draw nodes in 3D
node_sizes = 40  # Size of nodes

# Draw nodes in 3D with gradient colors
ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=node_colors, s=node_sizes)

# Step 6: Draw edges
for edge in G.edges():
    x_coords = [coords[edge[0], 0], coords[edge[1], 0]]
    y_coords = [coords[edge[0], 1], coords[edge[1], 1]]
    z_coords = [coords[edge[0], 2], coords[edge[1], 2]]
    ax.plot(x_coords, y_coords, z_coords, color='gray', alpha=0.5)  # Edges in gray

# Step 7: Set labels and title
ax.set_xlabel('X Coordinate (center_x)')
ax.set_ylabel('Y Coordinate (center_y)')
ax.set_zlabel('Z Coordinate (center_z)')
ax.set_title('3D Graph Visualization with Blue-to-Red Gradient Colors')

# Optional: Create a color bar for reference
sm = plt.cm.ScalarMappable(cmap=blue_red_cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])  # Only needed for older Matplotlib versions
plt.colorbar(sm, ax=ax, label='Ground Truth Value (0 to 1)')

# Show the plot
plt.show()

#%%

import torch
from torch_geometric.loader import DataLoader

train_size = int(0.8 * len(graph_data_list))  # 80% pour l'entraînement
test_size = len(graph_data_list) - train_size

train_graphs = graph_data_list[:train_size]
test_graphs = graph_data_list[train_size:]


# Create DataLoader for training and testing
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32)

import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch


class NodeGNNModel(torch.nn.Module):
    def __init__(self, num_node_features):
        super(NodeGNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, 1)  # Output a single value per node

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Pass features through GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # Output predictions for each node
        x = self.fc(x)  # Output shape will be [N_nodes, 1]
        return x.squeeze()

import torch

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate the mean squared error loss
        mse_loss = torch.nn.functional.mse_loss(inputs, targets, reduction='none')
        
        # Calculate the focal loss components
        focal_loss = self.alpha * (1 - torch.exp(-mse_loss)) ** self.gamma * mse_loss
        
        # Apply reduction method
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Initialize model, loss function, and optimizer
num_node_features = train_graphs[0].x.shape[1]  # Number of node features
model = NodeGNNModel(num_node_features)

# Use focal loss instead of MSE
criterion = FocalLoss(gamma=2.0, alpha=0.25)  # You can adjust gamma and alpha as needed
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Modify the training loop to include focal loss
loss_values = []
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)

        # Calculate weights for focal loss
        weights = torch.where(batch.y > 0.5, 1.0, 0.1)  # This line can be adjusted based on the use case
        loss = criterion(out, batch.y)
        weighted_loss = (loss * weights).mean()  # Apply weights

        weighted_loss.backward()
        optimizer.step()

        total_loss += weighted_loss.item()

    average_loss = total_loss / len(train_loader)
    loss_values.append(average_loss)
    print(f'Epoch {epoch + 1}, Loss: {average_loss:.4f}')

# Plotting the loss values
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Training Loss', color='blue')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Evaluation loop
model.eval()  # Set model to evaluation mode
total_loss = 0
with torch.no_grad():
    for batch in test_loader:
        out = model(batch)  # Forward pass
        loss = criterion(out, batch.y)  # Calculate loss for each node
        total_loss += loss.item()

print(f'Test Loss: {total_loss / len(test_loader):.4f}')

# Plotting the loss values
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Training Loss', color='blue')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# Evaluation loop
model.eval()  # Set model to evaluation mode
total_loss = 0
with torch.no_grad():
    for batch in test_loader:
        out = model(batch)  # Forward pass
        loss = criterion(out, batch.y)  # Calculate loss for each node
        total_loss += loss.item()

print(f'Test Loss: {total_loss / len(test_loader):.4f}')

#%%


first_graph = graph_data_list[1]
model.eval()
with torch.no_grad():
    node_predictions = model(first_graph)  

ground_truth = first_graph.y
node_predictions_np = node_predictions.numpy()
ground_truth_np = ground_truth.numpy()

import matplotlib.pyplot as plt

# Create a figure to plot the predictions against the ground truth
plt.figure(figsize=(12, 6))

# X-axis: Node indices
node_indices = range(len(node_predictions_np))

# Scatter plot of predictions and ground truth
plt.scatter(node_indices, ground_truth_np, color='blue', label='Ground Truth', alpha=0.5)
plt.scatter(node_indices, node_predictions_np, color='red', label='Predictions', alpha=0.5)

# Add titles and labels
plt.title('Node Predictions vs Ground Truth for First Graph')
plt.xlabel('Node Index')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()

#%%

import torch
import matplotlib.pyplot as plt

# Assuming first_graph is already defined and the model has been evaluated
first_graph = graph_data_list[3]
model.eval()

with torch.no_grad():
    node_predictions = model(first_graph)  # Get model predictions

ground_truth = first_graph.y  # Get ground truth values
node_predictions_np = node_predictions.numpy()  # Convert predictions to NumPy
ground_truth_np = ground_truth.numpy()  # Convert ground truth to NumPy

# Calculate the differences (absolute differences can be helpful)
differences = node_predictions_np - ground_truth_np  # or np.abs(node_predictions_np - ground_truth_np) for absolute differences

# Create a figure to plot the differences
plt.figure(figsize=(12, 6))

# X-axis: Node indices
node_indices = range(len(differences))

# Scatter plot of differences
plt.scatter(node_indices, differences, color='purple', label='Prediction - Ground Truth', alpha=0.6)

# Add horizontal line at y=0 for reference
plt.axhline(0, color='red', linestyle='--', linewidth=1)

# Add titles and labels
plt.title('Differences Between Node Predictions and Ground Truth for First Graph')
plt.xlabel('Node Index')
plt.ylabel('Difference (Predictions - Ground Truth)')
plt.legend()
plt.grid()
plt.show()

# Create a figure to plot the differences as a bar plot
plt.figure(figsize=(12, 6))

# X-axis: Node indices
node_indices = range(len(differences))

# Bar plot of differences
plt.bar(node_indices, differences, color='purple', label='Prediction - Ground Truth', alpha=0.6)

# Add horizontal line at y=0 for reference
plt.axhline(0, color='red', linestyle='--', linewidth=1)

# Add titles and labels
plt.title('Differences Between Node Predictions and Ground Truth for First Graph')
plt.xlabel('Node Index')
plt.ylabel('Difference (Predictions - Ground Truth)')
plt.legend()
plt.grid()
plt.show()
