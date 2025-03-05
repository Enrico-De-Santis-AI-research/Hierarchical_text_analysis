import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import multiprocessing
from multiprocessing import Pool, cpu_count
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map  # Optimized for multiprocessing

"""
# 📊 **Training a Neural Network for Feature Dependency Modeling**
### **Author:** Enrico De Santis  
### **Version:** 1.3  
### **Last Updated:** [Date]  
### **Dependencies:** NumPy, Pandas, Torch (PyTorch), SciPy, Seaborn, Matplotlib, Multiprocessing, tqdm, sklearn

---

## **🔍 Overview**
This script trains a **Neural Network (NN)** to **model feature dependencies** in a dataset of linguistic constraints.  
The trained model is later used to **regularize the Maximum Entropy Model** by enforcing structured feature dependencies.

---

## **🎯 Objectives**
1. **Load Dataset**  
   - Extracts **linguistic constraint features** from `fake_data_nextLevel_normalized.csv`.  
   - Converts numerical features into **PyTorch tensors** for efficient computation.

2. **Split Data into Training, Validation, and Testing Sets**  
   - Ensures the model is evaluated on **unseen data** to avoid overfitting.

3. **Train a Neural Network to Learn Pairwise Feature Dependencies**  
   - Uses a **fully connected feedforward NN**.
   - Predicts dependency **strength** between linguistic feature pairs.

4. **Optimize with Mini-Batch Training Using DataLoader**  
   - Implements **gradient-based optimization** with Adam optimizer.
   - Uses **early stopping** to avoid overfitting.

5. **Evaluate Model Performance on Test Set**  
   - Computes final **test loss** to assess generalization.

6. **Compute & Visualize Feature Dependencies in Parallel**  
   - Uses **multiprocessing** to speed up **dependency computation**.
   - Saves results as a **dependency matrix** and heatmap.

---

## **📌 Methodology**
### **Step 1: Load Dataset**
- Loads preprocessed dataset (`fake_data_nextLevel_normalized.csv`).
- **Extracts only numerical constraint features** (removes `Label` and `Text`).
- Converts features to **PyTorch tensors** and moves them to **GPU (if available)**.

### **Step 2: Split Data for Training, Validation, and Testing**
- Uses **80%-10%-10% split**:
  - **80% training**
  - **10% validation**
  - **10% test**
- Ensures **test set remains unseen** for final evaluation.

### **Step 3: Define Neural Network**
- **FeatureDependencyNN** has:
  - **3 fully connected layers**: `(2 → 32 → 16 → 1)`
  - **ReLU activations** for non-linearity.
  - Outputs a **dependency score** between two features.

### **Step 4: Parallelized Feature Dependency Computation**
- Generates **all feature pairs (i, j)** and **computes dependency targets**:
  - Input: `[mean(feature_i), mean(feature_j)]`
  - Target: `|mean(feature_i) - mean(feature_j)|`
- Uses **multiprocessing** to **speed up computations**.

### **Step 5: Train the Neural Network**
- Uses **mini-batch training** with `batch_size=32`.
- **Adam optimizer** updates model weights.
- **Mean Squared Error (MSE)** as loss function.

### **Step 6: Early Stopping with Validation Loss**
- Saves the **best model** based on **minimum validation loss**.
- **Stops training if validation loss doesn’t improve for 10 epochs**.

### **Step 7: Compute Feature Dependencies**
- Loads the **trained model** and applies it to all feature pairs.
- Uses **multiprocessing** for parallel dependency computation.
- Saves results as a **dependency matrix**.

### **Step 8: Generate & Save Dependency Heatmap**
- Saves results as:
  - `NN_Feature_Dependencies.csv`
  - `NN_Feature_Dependencies_Heatmap.png`
  - `NN_Feature_Dependencies_Heatmap.pdf`

---

## **📌 Main Parameters & Their Meaning**
| Parameter | Description |
|-----------|------------|
| `data_filename` | Dataset file (`fake_data_nextLevel_normalized.csv`). |
| `batch_size` | Number of feature pairs processed per training step. |
| `num_epochs` | Maximum training epochs (with early stopping). |
| `patience` | Number of epochs to wait before early stopping. |
| `model_path` | Path to save the **trained neural network**. |

---

## **📌 Outputs & File Structure**
| Output File | Description |
|------------|-------------|
| `Models/dependency_nn_best.pth` | Best-trained neural network (PyTorch). |
| `Results/NN_Training/NN_Feature_Dependencies.csv` | Feature dependency matrix. |
| `Results/NN_Training/NN_Feature_Dependencies_Heatmap.png` | Heatmap of dependencies. |

---

## **📌 How to Interpret the Results**
### **1️⃣ Test Loss**
| Test Loss | Interpretation |
|-----------|---------------|
| **< 0.01** | ✅ **Excellent** (highly structured dependencies). |
| **0.01 - 0.05** | ⚠️ **Moderate** (some dependencies detected). |
| **> 0.05** | ❌ **Weak** (poor feature dependency modeling). |

### **2️⃣ Heatmap Interpretation**
- **Darker red = Strong dependencies**.
- **Blue = Weak dependencies**.
- **Diagonal elements = 0** (self-dependencies are meaningless).

---

## **📌 Limitations**
1. **Assumes Feature Dependencies Are Pairwise**  
   - Does **not** model higher-order interactions (e.g., three-way dependencies).

2. **Feature Mean as Input Might Be Oversimplified**  
   - Could be replaced with **higher-order statistics (variance, entropy, etc.)**.

3. **Multiprocessing Limited by CPU Cores**  
   - If CPU cores are **too few**, speed-up is minimal.

---

"""

# ---- Set Start Method for CUDA Compatibility on Linux ----
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn", force=True)

# ---- Step 1: Load Dataset ----
data_dir = "Data"
data_filename = "fake_data_nextLevel_normalized.csv"
data_path = os.path.join(data_dir, data_filename)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Error: The dataset '{data_filename}' was not found in '{data_dir}/'. "
                            "Please check the filename or run 'generate_fake_data_nextLevel.py' first.")

print(f"✅ Loading dataset: {data_filename}")
df = pd.read_csv(data_path)

index_columns = [
    "Zipf_Law", "Kolmogorov_Complexity", "Syllable_Counts", "Word_Lengths",
    "Lexical_Diversity_TTR", "Lexical_Diversity_Shannon", "POS_Tag_Distribution",
    "Avg_Sentence_Length", "Dependency_Complexity", "Punctuation_Usage",
    "Readability_Dale_Chall", "Readability_Flesch_Kincaid", "Referential_Integrity",
    "Cosine_Similarity", "WordNet_Similarity", "Lexical_Chains", "LDA_Topic_Consistency",
    "Style_Consistency", "BLEU_Score", "Jaccard_Score", "ROUGE_Score", "Text_Flow_Analysis"
]
df = df[index_columns]

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

# Convert DataFrame to Torch tensor
features = torch.tensor(df.values, dtype=torch.float32).to(device)

# ---- Step 2: Split Data into Training, Validation, and Testing ----
train_features, temp_features = train_test_split(features, test_size=0.2, random_state=42)
val_features, test_features = train_test_split(temp_features, test_size=0.5, random_state=42)

# ---- Step 3: Define Neural Network for Feature Dependencies ----
class FeatureDependencyNN(nn.Module):
    def __init__(self):
        super(FeatureDependencyNN, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
dependency_model = FeatureDependencyNN().to(device)
optimizer = optim.Adam(dependency_model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# ---- Step 4: Parallelized Feature Dependency Computation ----
def prepare_feature_pairs(feature_set):
    feature_pairs, targets = [], []
    for i in range(feature_set.shape[1]):
        for j in range(i + 1, feature_set.shape[1]):
            pair = torch.tensor([feature_set[:, i].mean(), feature_set[:, j].mean()], dtype=torch.float32).to(device)
            target = torch.abs(feature_set[:, i].mean() - feature_set[:, j].mean()).to(device)
            feature_pairs.append(pair)
            targets.append(target)
    return torch.stack(feature_pairs), torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(device)

train_pairs, train_targets = prepare_feature_pairs(train_features)
val_pairs, val_targets = prepare_feature_pairs(val_features)
test_pairs, test_targets = prepare_feature_pairs(test_features)

# ---- Step 5: Mini-Batch Training Using DataLoader ----
batch_size = 32
train_loader = DataLoader(TensorDataset(train_pairs, train_targets), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_pairs, val_targets), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(test_pairs, test_targets), batch_size=batch_size)

# ---- Step 6: Train the Neural Network with Validation and Checkpoints ----
num_epochs = 500
best_val_loss = float("inf")
early_stopping_counter = 0
patience = 10

results_dir = "Results/NN_Training"
os.makedirs(results_dir, exist_ok=True)
model_dir = "Models"
os.makedirs(model_dir, exist_ok=True)

print("🔄 Training neural network...")
for epoch in range(num_epochs):
    dependency_model.train()
    train_loss = 0

    for batch_features, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = dependency_model(batch_features)
        loss = loss_function(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    dependency_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            val_outputs = dependency_model(batch_features)
            val_loss += loss_function(val_outputs, batch_targets).item()
    
    val_loss /= len(val_loader)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Save best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(dependency_model.state_dict(), os.path.join(model_dir, "dependency_nn_best.pth"))
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch}. No improvement in validation loss.")
            break

# ---- Step 7: Evaluate on Test Set ----
dependency_model.load_state_dict(torch.load(os.path.join(model_dir, "dependency_nn_best.pth"), weights_only=True))
dependency_model.eval()
test_loss = 0
with torch.no_grad():
    for batch_features, batch_targets in test_loader:
        test_outputs = dependency_model(batch_features)
        test_loss += loss_function(test_outputs, batch_targets).item()

test_loss /= len(test_loader)
print(f"✅ Test Loss: {test_loss:.6f}")

# ---- Step 8: Parallelized Feature Dependency Computation and Plotting ----
def compute_dependency(args):
    """Compute feature dependency score for a given feature pair (i, j)."""
    i, j = args
    pair = torch.tensor([features[:, i].mean(), features[:, j].mean()], dtype=torch.float32).unsqueeze(0).to(device)
    prediction = dependency_model(pair).item()
    return (i, j, prediction)

if __name__ == "__main__":  # Ensure multiprocessing runs correctly on Linux
    # Ensure correct multiprocessing start method
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn", force=True)

    num_cores = 2# min(cpu_count(), len(index_columns))  # Use available CPU cores but limit to feature count
    print(f"🔄 Computing feature dependencies using {num_cores} parallel processes...")

    # Prepare feature index pairs
    index_pairs = [(i, j) for i in range(len(index_columns)) for j in range(i + 1, len(index_columns))]

    # Use tqdm to display a progress bar while running multiprocessing
    results = process_map(compute_dependency, index_pairs, max_workers=num_cores, chunksize=5)

    # Store results in a matrix
    feature_dependency_matrix = torch.zeros((len(index_columns), len(index_columns)))
    for i, j, prediction in results:
        feature_dependency_matrix[i, j] = prediction
        feature_dependency_matrix[j, i] = prediction  # Symmetric matrix

    # Convert to DataFrame and Save
    dependency_df = pd.DataFrame(feature_dependency_matrix.cpu().numpy(), index=index_columns, columns=index_columns)
    dependency_df.to_csv(os.path.join(results_dir, "NN_Feature_Dependencies.csv"))

    # Plot Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(dependency_df, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5)
    plt.title("Neural Network Learned Feature Dependencies")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "NN_Feature_Dependencies_Heatmap.png"), dpi=300)
    plt.savefig(os.path.join(results_dir, "NN_Feature_Dependencies_Heatmap.pdf"), dpi=300)
    plt.show()

    print(f"✅ All results saved in {results_dir}/")