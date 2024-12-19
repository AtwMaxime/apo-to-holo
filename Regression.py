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
        node_features = node_df[['atom_index', 'atom_type', 'radius', 
                'voromqa_sas_potential', 'residue_mean_sas_potential', 
                'residue_size', 'sas_area', 'voromqa_sas_energy', 
                'voromqa_score_a', 'volume', 'ev28', 'ev56']].values
        
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        ground_truth = node_df['ground_truth'].values  # Extract ground truth
        
        # Step 4: Create a Data object
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        
        data.y = torch.tensor(ground_truth, dtype=torch.float)
        
        # Step 5: Store the Data object in the list
        graph_data_list.append(data)

# Now, you have all the graphs in the list `graph_data_list` as PyTorch Geometric Data objects
print(f"Loaded {len(graph_data_list)} graphs.")

import matplotlib.pyplot as plt

# Initialiser une liste pour collecter toutes les valeurs de ground truth
all_ground_truth_values = []

# Itérer à travers chaque graphique dans graph_data_list
for graph in graph_data_list:
    # Itérer à travers chaque nœud dans le graphique pour obtenir la valeur de ground truth
    for node_index in range(graph.y.shape[0]):  # graph.y contient le ground truth pour chaque nœud
        ground_truth_value = graph.y[node_index].item()  # Obtenir la valeur de ground truth
        all_ground_truth_values.append(ground_truth_value)  # L'ajouter à la liste


# Plotter la distribution des valeurs de ground truth à travers tous les nœuds
plt.figure(figsize=(12, 6))
plt.hist(all_ground_truth_values, bins=50, color='blue', alpha=0.7, edgecolor='black')  # Utiliser 50 bins
plt.title('Distribution des valeurs de Ground Truth à travers tous les nœuds dans tous les graphes')
plt.xlabel('Valeur de Ground Truth')
plt.ylabel('Fréquence')
plt.xlim(0, 1)  # En supposant que les valeurs de ground truth sont comprises entre 0 et 1
plt.grid(True)

# Afficher le plot
plt.tight_layout()
plt.show()


import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde

# Assuming all_ground_truth_values is a numpy array or list of ground truth values
# Example: all_ground_truth_values = np.random.rand(10000)  # Replace with your actual values.

# Step 1: Estimate the PDF using Gaussian KDE
def calculate_weights(all_ground_truth_values):
    # Create a KDE for the ground truth values
    kde = gaussian_kde(all_ground_truth_values)
    
    # Generate values for which to evaluate the KDE
    x = np.linspace(0, 1, 100)  # Evaluate from 0 to 1
    pdf_values = kde(x)  # PDF values at these points
    
    # Step 2: Calculate weights as the inverse of the PDF
    weights = 1.0 / (pdf_values + 1e-5)  # Avoid division by zero with small constant
    
    # Normalize the weights to ensure they sum to 1 (optional)
    weights /= np.sum(weights)

    return x, weights

# Call the function to get weights
x, weights = calculate_weights(all_ground_truth_values)

# Step 3: Create a mapping from ground truth values to their corresponding weights
def create_weight_mapping(x, weights):
    weight_mapping = {x_val: weight for x_val, weight in zip(x, weights)}
    return weight_mapping

# Create the weight mapping
weight_mapping = create_weight_mapping(x, weights)

# Example of how to retrieve weights during loss computation
def get_weight_for_target(target):
    # Use interpolation to find the weight for the continuous target value
    return np.interp(target, x, weights)

# Now, integrate this into your loss function
class WeightedMSELoss(nn.Module):
    def __init__(self, weight_mapping):
        super(WeightedMSELoss, self).__init__()
        self.weight_mapping = weight_mapping

    def forward(self, predictions, targets):
        # Get the weights for the targets using interpolation
        target_weights = torch.tensor([get_weight_for_target(float(t)) for t in targets.cpu().numpy()]).to(predictions.device)
        
        mse_loss = (predictions - targets) ** 2
        weighted_loss = target_weights * mse_loss
        return weighted_loss.mean()  # Average the loss


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

import os
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from torch_geometric.nn import GATConv

class NodeGNNModel(torch.nn.Module):
    def __init__(self, num_node_features):
        super(NodeGNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)  # First GCN layer
        self.conv2 = GCNConv(64, 32)  # Second GCN layer
        self.fc = torch.nn.Linear(32, 1)  # Output a single value per node

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))  # Apply the first GCN layer
        x = F.relu(self.conv2(x, edge_index))  # Apply the second GCN layer
        x = self.fc(x)  # Fully connected layer
        x = torch.sigmoid(x)  # Apply sigmoid to the output
        return x.squeeze()

# Define the Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction  # Reduction method

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Probability of the true class

        # Compute focal loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Initialize the dataset
graph_folder = r"apo"
graph_data_list = []

for filename in os.listdir(graph_folder):
    if filename.endswith("_links.csv"):
        node_filename = filename.replace("_links.csv", "_nodes.csv")
        link_file_path = os.path.join(graph_folder, filename)
        node_file_path = os.path.join(graph_folder, node_filename)

        edge_df = pd.read_csv(link_file_path)
        edge_index = torch.tensor(edge_df[['atom_index1', 'atom_index2']].values.T, dtype=torch.long)
        edge_attr = torch.tensor(edge_df[[]].values, dtype=torch.float)
        node_df = pd.read_csv(node_file_path)
        node_features = torch.tensor(node_df[['ground_truth']].values, dtype=torch.float)
        ground_truth = torch.tensor(node_df['ground_truth'].values, dtype=torch.float)
        
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=ground_truth)
        graph_data_list.append(data)

print(f"Loaded {len(graph_data_list)} graphs.")

# K-Fold Cross-Validation Setup
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
loss_values_per_fold = []

# Training and Validation Loop
num_epochs = 20

for fold, (train_idx, test_idx) in enumerate(kf.split(graph_data_list)):
    print(f"Starting Fold {fold + 1}/{k_folds}...")
    
    # Split the dataset into train and test graphs
    train_graphs = [graph_data_list[i] for i in train_idx]
    test_graphs = [graph_data_list[i] for i in test_idx]
    
    # Create DataLoader for training and testing
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32)
    
    # Reinitialize model, optimizer, and loss function for each fold
    num_node_features = train_graphs[0].x.shape[1]
    model = NodeGNNModel(num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # criterion = FocalLoss(alpha=0.5, gamma=2)
    # criterion = WeightedMSELoss(weight_mapping)
    criterion = torch.nn.MSELoss()
    
    # Track losses for this fold
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_losses.append(total_train_loss)
        
        # Validation loss
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                val_out = model(batch)
                val_loss = criterion(val_out, batch.y)
                total_val_loss += val_loss.item()
        val_losses.append(total_val_loss)
        
        print(f"Epoch {epoch + 1}, Training Loss: {total_train_loss:.4f}, Validation Loss: {total_val_loss:.4f}")
    
    # Save model for this fold
    torch.save(model.state_dict(), f"model_fold_{fold + 1}.pth")
    
    # Store losses for this fold
    loss_values_per_fold.append((train_losses, val_losses))
    
    # Evaluate all test graphs
    for idx, test_graph in enumerate(test_graphs):
        model.eval()
        with torch.no_grad():
            node_predictions = model(test_graph)
        ground_truth = test_graph.y
        node_predictions_np = node_predictions.view(-1).numpy()
        ground_truth_np = ground_truth.numpy()
        differences = node_predictions_np - ground_truth_np
        
        if idx ==0:
            node_indices = range(len(node_predictions_np))
            # Plot predictions vs ground truth
            # plt.figure(figsize=(12, 6))
            # plt.scatter(node_indices, ground_truth_np, color='blue', label='Ground Truth', alpha=0.5)
            # plt.scatter(node_indices, node_predictions_np, color='red', label='Predictions', alpha=0.5)
            # plt.title(f'Node Predictions vs Ground Truth (Fold {fold + 1}, Graph {idx + 1})')
            # plt.xlabel('Node Index')
            # plt.ylabel('Value')
            # plt.legend()
            # plt.grid()
            # plt.show()
            
            # Plot differences as scatter plot
            plt.figure(figsize=(12, 6))
            plt.scatter(node_indices, differences, color='purple', label='Prediction - Ground Truth', alpha=0.6)
            plt.axhline(0, color='red', linestyle='--', linewidth=1)
            plt.title(f'Differences Between Node Predictions and Ground Truth (Fold {fold + 1}, Graph {idx + 1})')
            plt.xlabel('Node Index')
            plt.ylabel('Difference')
            plt.legend()
            plt.grid()
            plt.show()

# Plot Training and Validation Loss for Each Fold
plt.figure(figsize=(10, 5))
for fold, (train_losses, val_losses) in enumerate(loss_values_per_fold):
    plt.plot(train_losses, label=f'Fold {fold + 1} Training Loss')
    plt.plot(val_losses, label=f'Fold {fold + 1} Validation Loss')
plt.title('Training and Validation Loss Across Folds')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()




#%%


def predict(models_folder, test_graph):
    """
    Use the 5 trained models to predict on a test graph using ensemble averaging.

    Args:
        models_folder (str): Folder containing the saved model files.
        test_graph (torch_geometric.data.Data): Test graph for prediction.

    Returns:
        numpy.ndarray: Averaged predictions from all models.
    """
    # Initialize an empty list to store predictions from each model
    predictions_list = []
    
    for fold in range(1, 6):  # Assuming 5 folds
        # Load the model structure
        num_node_features = test_graph.x.shape[1]
        model = NodeGNNModel(num_node_features)
        
        # Load the saved weights
        model_path = os.path.join(models_folder, f"model_fold_{fold}.pth")
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set model to evaluation mode
        
        # Make predictions
        with torch.no_grad():
            predictions = model(test_graph)
            predictions_list.append(predictions.numpy())
    
    # Perform ensemble averaging
    predictions_ensemble = np.mean(predictions_list, axis=0)
    return predictions_ensemble

predictions = predict("",graph_data_list[1])

ground_truth_np = graph_data_list[1].y.numpy()
# Calculate the differences (residuals)
differences = predictions - ground_truth_np

# X-axis: Node indices
node_indices = range(len(differences))

# Create a bar plot of the differences
plt.figure(figsize=(12, 6))
plt.bar(node_indices, differences, color='purple', alpha=0.6, label='Prediction - Ground Truth')

# Add a horizontal line at y=0 for reference
plt.axhline(0, color='red', linestyle='--', linewidth=1)

# Add titles and labels
plt.title('Residuals (Prediction - Ground Truth) for Nodes')
plt.xlabel('Node Index')
plt.ylabel('Difference')
plt.legend()
plt.grid()
plt.show()

