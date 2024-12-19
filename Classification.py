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


# Define the GNN model for classification
class NodeGNNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(NodeGNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 512)  # First GCN layer
        self.conv2 = GCNConv(512, 128)  # Second GCN layer
        self.fc = torch.nn.Linear(128, num_classes)  # Output classes for classification

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))  # Apply the first GCN layer
        x = F.relu(self.conv2(x, edge_index))  # Apply the second GCN layer
        x = self.fc(x)  # Fully connected layer
        return F.log_softmax(x, dim=1)  # Apply log softmax for classification


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
        node_features = torch.tensor(node_df[['atom_index', 'atom_type', 'radius', 
                'voromqa_sas_potential', 'residue_mean_sas_potential', 
                'residue_size', 'sas_area', 'voromqa_sas_energy', 
                'voromqa_score_a', 'volume', 'ev28', 'ev56','ground_truth']].values, dtype=torch.float)

        # Convert ground truth values to class labels
        ground_truth = node_df['ground_truth'].values
        class_labels = np.digitize(ground_truth, bins=[0, 0.3, 0.5, 0.7, 1]) - 1  # Bins for [0, 0.3), [0.3, 0.5), [0.5, 0.7), [0.7, 1]
        class_labels[class_labels == 4] = 3
        class_labels[class_labels < 0] = 0  # Clamp any out-of-bound lower labels to 0
        ground_truth = torch.tensor(class_labels, dtype=torch.long)  # Class labels as LongTensor
        
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=ground_truth)
        graph_data_list.append(data)

print(f"Loaded {len(graph_data_list)} graphs.")

# K-Fold Cross-Validation Setup
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
loss_values_per_fold = []

# Training and Validation Loop
num_epochs = 100
num_classes = 4  # Number of classes

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
    model = NodeGNNModel(num_node_features, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    class0=32453
    class1=4757
    class2=1282
    class3=423
    total=class0 + class1 + class2 + class3
    weight_class0 = total / (4 * class0)
    weight_class1 = total / (4 * class1)
    weight_class2 = total / (4 * class2)
    weight_class3 = total / (4 * class3)
    
    print(weight_class0,weight_class1,weight_class2,weight_class3)
    
    
    class_weights = torch.tensor([weight_class0, weight_class1, weight_class2, weight_class3])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    
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
            loss = criterion(out, batch.y)  # Compute classification loss
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
                val_loss = criterion(val_out, batch.y)  # Validation loss
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
            predicted_classes = torch.argmax(node_predictions, dim=1)  # Get predicted class labels
        ground_truth = test_graph.y
        
        # Visualization or additional evaluation code can go here

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

import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def predict_and_evaluate(models_folder, graph_data_list):
    """
    Load each model, predict classes for each node in each graph, 
    calculate accuracy for each class, and display the confusion matrix.

    Args:
        models_folder (str): Folder containing the saved model files.
        graph_data_list (list): List of graphs for prediction.

    Returns:
        dict: Dictionary containing accuracies for each class and confusion matrix.
    """
    num_classes = 4  # Assuming we have 4 classes: 0, 1, 2, 3
    overall_predictions = []
    overall_ground_truths = []

    # Loop through each graph in the graph_data_list
    for graph_data in graph_data_list:
        # Initialize a list to store predictions for the current graph
        graph_predictions = []

        # Loop through each fold/model
        for fold in range(1, 6):  # Assuming 5 folds
            # Load the model structure
            num_node_features = graph_data.x.shape[1]
            model = NodeGNNModel(num_node_features,num_classes)

            # Load the saved weights
            model_path = os.path.join(models_folder, f"model_fold_{fold}.pth")
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set model to evaluation mode

            # Make predictions
            with torch.no_grad():
                predictions = model(graph_data)
                graph_predictions.append(predictions.numpy())

        # Perform ensemble averaging
        predictions_ensemble = np.mean(graph_predictions, axis=0)

        # Get predicted classes by taking argmax
        predicted_classes = np.argmax(predictions_ensemble, axis=1)

        # Ground truth
        ground_truth_np = graph_data.y.numpy()
        
        # Append predictions and ground truth
        overall_predictions.extend(predicted_classes)
        overall_ground_truths.extend(ground_truth_np)

    # Calculate accuracy for each class
    overall_predictions = np.array(overall_predictions)
    overall_ground_truths = np.array(overall_ground_truths)

    accuracies = {}
    for cls in range(num_classes):  # Assuming classes 0, 1, 2, 3
        class_indices = np.where(overall_ground_truths == cls)[0]
        if len(class_indices) > 0:
            class_accuracy = np.mean(overall_predictions[class_indices] == cls)
            accuracies[f'Class {cls}'] = class_accuracy
        else:
            accuracies[f'Class {cls}'] = None  # No instances of this class

    # Calculate the confusion matrix
    cm = confusion_matrix(overall_ground_truths, overall_predictions, labels=range(num_classes))

    return accuracies, cm

# Call the function to predict and evaluate on all graphs in graph_data_list
models_folder = ""  # Path to your models folder
accuracies, confusion_mat = predict_and_evaluate(models_folder, graph_data_list)

# Display the accuracies for each class
for cls, accuracy in accuracies.items():
    if accuracy is not None:
        print(f"{cls}: Accuracy = {accuracy:.2f}")
    else:
        print(f"{cls}: No samples available.")

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=range(4), yticklabels=range(4))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()