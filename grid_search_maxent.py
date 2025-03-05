import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.optimize as opt
import os
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from multiprocessing import Pool, get_start_method, set_start_method

"""
# 📊 **Parallel Grid Search for Maximum Entropy Model with Neural Dependency Regularization**
### **Author:** Enrico De Santis  
### **Version:** 1.2  
### **Last Updated:** [Date]  
### **Dependencies:** NumPy, Pandas, Torch (PyTorch), SciPy, Seaborn, Matplotlib, Multiprocessing, tqdm

---

## **🔍 Overview**
This script performs a **parallelized grid search** for hyperparameter tuning of a **Maximum Entropy Model** with **neural dependency regularization**.  
The objective is to find the **best α (L2 regularization) and β (neural dependency penalty)** that minimize **constraint deviation loss** while maintaining structured dependencies between linguistic features.

---

## **🎯 Objectives**
1. **Load Dataset**  
   - Extracts **linguistic constraint features** from `fake_data_nextLevel_normalized.csv`.  
   - Converts numerical features into numpy format for fast computation.

2. **Load Pre-trained Neural Network for Feature Dependency Regularization**  
   - The **FeatureDependencyNN** predicts dependency strength between pairs of linguistic features.
   - Adds **dependency-based regularization** to the optimization process.

3. **Compute Linguistic Constraints Based on User-Selected Type**  
   - Options: `mean`, `variance`, `skewness`, `kurtosis`.
   - Computes empirical constraints accordingly.

4. **Train the Maximum Entropy Model Using L-BFGS-B Optimization**  
   - The objective function minimizes the difference between **expected and empirical constraints**.
   - Enforces **structured dependencies between linguistic features**.

5. **Perform Grid Search in Parallel to Tune Hyperparameters**  
   - Uses **multiprocessing** to speed up hyperparameter search.
   - Evaluates multiple combinations of **α (L2 regularization) and β (dependency penalty)**.

6. **Save & Visualize Grid Search Results**  
   - Saves **best hyperparameters** and corresponding loss values.
   - Generates **heatmap visualization** of grid search results.

---

## **📌 Methodology**
### **Step 1: Load Dataset**
- Loads preprocessed dataset (`fake_data_nextLevel_normalized.csv`).
- **Extracts only numerical constraint features** (removes `Label` and `Text`).
- Converts features to **numpy arrays** for optimization.

### **Step 2: Load Pre-trained Neural Network**
- **FeatureDependencyNN** is a **feedforward neural network** trained to model **pairwise dependencies**.
- **Regularizes λ values** based on known feature relationships.

### **Step 3: Compute Linguistic Constraints**
- Supports multiple constraint types:
  - **Mean**: Average value per feature.
  - **Variance**: Measures dispersion.
  - **Skewness**: Measures asymmetry.
  - **Kurtosis**: Measures tail heaviness.

### **Step 4: Define Maximum Entropy Loss Function**
The optimization objective is:

\[
\mathcal{L} = \sum (E_{\text{model}}[f] - E_{\text{data}}[f])^2 + \alpha ||\lambda||^2 + \beta \sum_{\text{pairs}} \text{NN}(x_i, x_j) \cdot (\lambda_i - \lambda_j)^2
\]

Where:
- **First term:** Minimizes constraint deviations.
- **Second term (α L2 regularization):** Prevents overfitting.
- **Third term (β NN regularization):** Penalizes λ values that violate known dependencies.

### **Step 5: Run Grid Search Using Parallel Processing**
- Generates all combinations of:
  - `α (L2 regularization) ∈ {0.001, 0.01, 0.1, 1.0}`
  - `β (neural dependency weight) ∈ {0.001, 0.01, 0.1, 1.0}`
- Uses **multiprocessing.Pool** to evaluate all experiments in parallel.
- Saves **best (α, β) pair** that minimizes loss.

### **Step 6: Save & Visualize Results**
- Saves trained **grid search results** in `Results/Grid_Search_[constraint_type]/`.
- Generates **heatmap of loss values**.

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
| `Results/Grid_Search_[constraint_type]/Grid_Search_Results.csv` | Stores α, β values and corresponding losses. |
| `Results/Grid_Search_[constraint_type]/Grid_Search_Loss_Heatmap.png` | Heatmap visualization of loss values. |

---

## **📌 How to Interpret the Results**
### **1️⃣ Generalization Performance**
| Test Loss | Interpretation |
|-----------|---------------|
| **< 0.05** | ✅ **Excellent generalization** (model works well on unseen data). |
| **0.05 - 0.2** | ⚠️ **Moderate generalization** (some risk of overfitting). |
| **> 0.2** | ❌ **High overfitting risk** (model may not generalize well). |

### **2️⃣ Best Hyperparameter Selection**
- **Optimal α (L2 regularization)** balances regularization strength.
- **Optimal β (dependency penalty)** ensures dependency constraints are respected.

### **3️⃣ Visualizations**
- **Heatmap of loss values** across α and β choices.

---

## **📌 Limitations**
1. **Grid Search is Computationally Expensive**  
   - Scaling α and β values **increases the number of experiments**.
   - Consider **Bayesian Optimization** for better efficiency.

2. **Pairwise Dependencies Only**  
   - Assumes **feature dependencies are pairwise**, ignoring higher-order interactions.
  
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

# ---- Step 2: Load Pre-trained Neural Network ----
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

# ---- Step 3: Define Constraint Selection Function ----
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

# ---- Step 4: Define Maximum Entropy Loss Function ----
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

# ---- Step 5: Define Grid Search Function ----
def run_experiment(params):
    alpha, beta = params
    lambda_init = np.random.uniform(-0.1, 0.1, df.shape[1])

    opt_result = opt.minimize(
        max_entropy_loss,
        lambda_init,
        args=(features, constraints, alpha, beta, dependency_model),
        method="L-BFGS-B"
    )
    
    return alpha, beta, opt_result.fun  # Return alpha, beta, and final loss value

# ---- Step 6: Define Hyperparameter Grid ----
alpha_values = [0.001, 0.01, 0.1, 1.0]  # Regularization strength
beta_values = [0.001, 0.01, 0.1, 1.0]  # NN Dependency weight

hyperparameter_grid = list(product(alpha_values, beta_values))

# ---- Step 7: Run Grid Search in Parallel ----
if __name__ == "__main__":
    if get_start_method(allow_none=True) is None:
        set_start_method("spawn", force=True)

    num_cores = min(multiprocessing.cpu_count(), len(hyperparameter_grid))
    print(f"🔄 Running grid search with {num_cores} parallel workers...")

    with Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(run_experiment, hyperparameter_grid), total=len(hyperparameter_grid)))

    # Store results
    results_df = pd.DataFrame(results, columns=["Alpha", "Beta", "Loss"])
    best_params = results_df.loc[results_df["Loss"].idxmin()]

    # Save results
    results_dir = f"Results/Grid_Search_{CONSTRAINT_TYPE}"
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, "Grid_Search_Results.csv"), index=False)

    print(f"✅ Best Hyperparameters: Alpha = {best_params['Alpha']}, Beta = {best_params['Beta']}")
    print(f"✅ Loss: {best_params['Loss']}")

    # ---- Step 8: Generate Heatmap of Loss Values ----
    results_pivot = results_df.pivot(index="Alpha", columns="Beta", values="Loss")

    plt.figure(figsize=(10, 6))
    sns.heatmap(results_pivot, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5)
    plt.title(f"Grid Search Loss Values ({CONSTRAINT_TYPE.capitalize()})")
    plt.xlabel("Beta (NN Dependency Weight)")
    plt.ylabel("Alpha (Regularization Strength)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "Grid_Search_Loss_Heatmap.png"), dpi=300)
    plt.savefig(os.path.join(results_dir, "Grid_Search_Loss_Heatmap.pdf"), dpi=300)
    plt.show()

    print(f"✅ Grid Search results and plots saved in {results_dir}/")
