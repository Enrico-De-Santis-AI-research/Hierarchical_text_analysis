import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.optimize as opt
import os
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import get_start_method, set_start_method

"""
# 📊 **Maximum Entropy Model Training with Neural Dependency Regularization**
### **Author:** Enrico De Santis 
### **Version:** 1.1  
### **Last Updated:** [Date]  
### **Dependencies:** NumPy, Pandas, Torch (PyTorch), SciPy, Seaborn, Matplotlib, Multiprocessing  

---

## **🔍 Overview**
This script **trains a Maximum Entropy Model** with **linguistic constraints** and **neural dependency regularization**. It incorporates a **pre-trained neural network (NN) model** to **regularize feature dependencies**, ensuring that estimated **Lagrange multipliers (λ values)** respect known hierarchical relationships between linguistic features.

---

## **🎯 Objectives**
1. **Load the Dataset**  
   - Extracts **linguistic constraint features** from `fake_data_nextLevel_normalized.csv`.  
   - Ensures correct data type for numerical processing.

2. **Select and Compute Linguistic Constraints**  
   - Allows user to select constraint type (`mean`, `variance`, `skewness`, `kurtosis`).
   - Computes empirical constraints accordingly.

3. **Load Optimal Hyperparameters from Grid Search**  
   - Retrieves **best α (L2 regularization) and β (neural dependency penalty)** values.
   - Ensures training is performed under optimal settings.

4. **Load a Pre-trained Neural Network for Feature Dependency Regularization**  
   - The **FeatureDependencyNN** predicts dependency strength between pairs of linguistic features.
   - Adds **dependency-based regularization** to the optimization process.

5. **Train the Maximum Entropy Model Using L-BFGS-B Optimization**  
   - The objective function minimizes the difference between **expected and empirical constraints**.
   - Enforces **structured dependencies between linguistic features**.

6. **Save & Visualize Lagrange Multipliers (λ Values)**  
   - Stores **trained λ values** for further interpretation.
   - Generates **bar plots** showing constraint strengths across different linguistic levels.

---

## **📌 Methodology**
### **Step 1: Load Dataset**
- Loads preprocessed dataset (`fake_data_nextLevel_normalized.csv`).
- **Extracts only numerical constraint features** (removes `Label` and `Text`).
- Converts features to **numpy arrays** for optimization.

### **Step 2: Compute Constraints**
- Supports multiple constraint types:
  - **Mean**: Average value per feature.
  - **Variance**: Measures dispersion.
  - **Skewness**: Measures asymmetry.
  - **Kurtosis**: Measures tail heaviness.

### **Step 3: Load Best Hyperparameters**
- Reads **best α (L2 regularization) and β (dependency penalty)** from previous `grid_search_maxent.py` results.

### **Step 4: Load Pre-trained Neural Network**
- **FeatureDependencyNN** is a **feedforward neural network** trained to model **pairwise dependencies**.
- **Regularizes λ values** based on known feature relationships.

### **Step 5: Define Maximum Entropy Loss Function**
The optimization objective is:

\[
\mathcal{L} = \sum (E_{\text{model}}[f] - E_{\text{data}}[f])^2 + \alpha ||\lambda||^2 + \beta \sum_{\text{pairs}} \text{NN}(x_i, x_j) \cdot (\lambda_i - \lambda_j)^2
\]

Where:
- **First term:** Minimizes constraint deviations.
- **Second term (α L2 regularization):** Prevents overfitting.
- **Third term (β NN regularization):** Penalizes λ values that violate known dependencies.

### **Step 6: Train the Maximum Entropy Model**
- Uses **L-BFGS-B** optimization to estimate **λ values**.
- Minimizes constraint violations while respecting **hierarchical dependencies**.

### **Step 7: Save & Interpret Results**
- Saves trained **λ values** in `Results/Trained_MaxEnt_[constraint_type]/`.
- Generates **bar plots** showing constraint strengths.

---

## **📌 Main Parameters & Their Meaning**
| Parameter | Description |
|-----------|------------|
| `data_filename` | Dataset file (`fake_data_nextLevel_normalized.csv`). |
| `CONSTRAINT_TYPE` | Defines how constraints are computed (`mean`, `variance`, `skewness`, `kurtosis`). |
| `best_alpha` | L2 regularization strength (from grid search). |
| `best_beta` | Neural dependency penalty (from grid search). |
| `model_path` | Path to pre-trained **FeatureDependencyNN**. |
| `lambda_init` | Initial Lagrange multipliers (randomized between -0.1 and 0.1). |

---

## **📌 Outputs & File Structure**
| Output File | Description |
|------------|-------------|
| `Results/Trained_MaxEnt_[constraint_type]/Trained_Lagrange_Multipliers.csv` | Stores trained λ values. |
| `Results/Trained_MaxEnt_[constraint_type]/Lagrange_Multipliers.png` | Barplot of constraint strengths per level. |

---

## **📌 How to Interpret the Results**
### **1️⃣ Generalization Performance**
| Test Loss | Interpretation |
|-----------|---------------|
| **< 0.05** | ✅ **Excellent generalization** (model works well on unseen data). |
| **0.05 - 0.2** | ⚠️ **Moderate generalization** (some risk of overfitting). |
| **> 0.2** | ❌ **High overfitting risk** (model may not generalize well). |

### **2️⃣ Constraint Strength Across Levels**
- **If lower levels (character, morphology, syntax) have higher λ values:**  
  ✅ *Supports hypothesis that lower levels impose stronger constraints.*  
- **If higher levels (semantics, stylistics) have unexpectedly strong constraints:**  
  ❌ *Suggests conceptual constraints may have a greater impact than expected.*

### **3️⃣ Visualizations**
- **Barplot:** Highlights constraint strengths per linguistic level.

---

## **📌 Limitations**
1. **Pairwise Dependencies Only**  
   - Assumes **feature dependencies are pairwise**, ignoring higher-order interactions.
  
2. **Neural Network Regularization Complexity**  
   - Regularization strength **(β value)** significantly impacts results.
   - Needs careful tuning to balance λ constraints and dependencies.

3. **Constraint Type Selection Affects Outcome**  
   - Different constraint types (`mean`, `variance`, `skewness`, `kurtosis`) produce **different λ distributions**.

---

"""


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

features = df.values  # Convert DataFrame to numpy

# ---- Step 2: Select Constraint Type ----
CONSTRAINT_TYPE = "mean"  # Options: "mean", "variance", "skewness", "kurtosis"

def compute_constraints(df, constraint_type):
    """Compute constraint values based on the selected method."""
    if constraint_type == "mean":
        return df.mean().values
    elif constraint_type == "variance":
        return df.var().values
    elif constraint_type == "skewness":
        return df.skew().values
    elif constraint_type == "kurtosis":
        return df.kurtosis().values
    else:
        raise ValueError(f"❌ Error: Invalid constraint type '{constraint_type}' selected!")

constraints = compute_constraints(df, CONSTRAINT_TYPE)

# ---- Step 3: Load Best Hyperparameters from Grid Search ----
grid_search_results_path = f"Results/Grid_Search_{CONSTRAINT_TYPE}/Grid_Search_Results.csv"

if not os.path.exists(grid_search_results_path):
    raise FileNotFoundError(f"❌ Error: Grid search results not found in {grid_search_results_path}. "
                            "Run 'grid_search_maxent.py' first.")

grid_results = pd.read_csv(grid_search_results_path)
best_params = grid_results.loc[grid_results["Loss"].idxmin()]
best_alpha, best_beta = best_params["Alpha"], best_params["Beta"]

print(f"✅ Using Best Hyperparameters: Alpha = {best_alpha}, Beta = {best_beta}")

# ---- Step 4: Load Pre-trained Neural Network ----
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

model_path = "Models/dependency_nn_best.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Error: The trained NN model '{model_path}' was not found. "
                            "Please run 'train_dependency_nn.py' first.")

dependency_model = FeatureDependencyNN().to(device)
dependency_model.load_state_dict(torch.load(model_path, weights_only=True))
dependency_model.eval()
print(f"✅ Loaded trained model from {model_path}")

# ---- Step 5: Define Maximum Entropy Loss Function ----
def max_entropy_loss(lambda_vals, features, constraints, alpha, beta, nn_model):
    exp_term = np.exp(-np.dot(features, lambda_vals))
    Z = np.sum(exp_term)
    p_x = exp_term / Z
    expected_constraints = np.dot(p_x, features)
    
    loss = np.sum((expected_constraints - constraints) ** 2)  # Constraint deviation
    
    # Regularization term
    loss += alpha * np.sum(lambda_vals ** 2)

    # NN Dependency Regularization
    nn_dependency_penalty = 0
    for i in range(len(lambda_vals)):
        for j in range(i + 1, len(lambda_vals)):  
            nn_input = torch.tensor([features[:, i].mean(), features[:, j].mean()], dtype=torch.float32).unsqueeze(0).to(device)
            nn_prediction = nn_model(nn_input).item()
            nn_dependency_penalty += nn_prediction * (lambda_vals[i] - lambda_vals[j]) ** 2

    loss += beta * nn_dependency_penalty
    return loss

# ---- Step 6: Train Maximum Entropy Model with Best Hyperparameters ----
lambda_init = np.random.uniform(-0.1, 0.1, df.shape[1])

opt_result = opt.minimize(
    max_entropy_loss,
    lambda_init,
    args=(features, constraints, best_alpha, best_beta, dependency_model),
    method="L-BFGS-B"
)

lambda_values = opt_result.x

# ---- Step 7: Save Results ----
results_dir = f"Results/Trained_MaxEnt_{CONSTRAINT_TYPE}"
os.makedirs(results_dir, exist_ok=True)

lambda_df = pd.DataFrame({"Index": index_columns, "Lambda": lambda_values})
lambda_df.to_csv(os.path.join(results_dir, "Trained_Lagrange_Multipliers.csv"), index=False)

print(f"✅ Maximum Entropy Model trained with best hyperparameters.")
print(f"✅ Results saved in {results_dir}/")

# ---- Step 8: Plot Lagrange Multipliers ----
plt.figure(figsize=(12, 6))
sns.barplot(x=lambda_df["Index"], y=lambda_df["Lambda"], color='royalblue')
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.title(f"Lagrange Multipliers (Trained with Best Hyperparameters) - {CONSTRAINT_TYPE.capitalize()}")
plt.ylabel("Lambda Value")
plt.xlabel("Linguistic Feature")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Lagrange_Multipliers.png"), dpi=300)
plt.savefig(os.path.join(results_dir, "Lagrange_Multipliers.pdf"), dpi=300)
plt.show()
