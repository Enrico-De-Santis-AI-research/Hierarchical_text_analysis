import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.optimize as opt
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
# 📊 **Maximum Entropy Model with Neural Network-Based Feature Dependency Regularization**
### **Author:** Enrico De Santis  
### **Version:** 1.4  
### **Last Updated:** [Date]  
### **Dependencies:** NumPy, Pandas, Torch, SciPy, Seaborn, Matplotlib, OS  

---

## **🔍 Overview**
This script implements a **Maximum Entropy (MaxEnt) Model** that is **regularized using a neural network (NN)**  
to **capture nonlinear feature dependencies**. The goal is to **optimize linguistic constraints**  
by integrating **a trained neural network model** that learns **feature interdependencies**.

---

## **🎯 Objectives**
1. **Load Dataset**  
   - Extracts **linguistic constraint features** from `fake_data_nextLevel_normalized.csv`.  
   - Keeps **only numerical indices**, ignoring `Label` and `Text`.  

2. **Compute Linguistic Constraints**  
   - Supports different **constraint types**:
     - `mean`: Average feature value.
     - `variance`: Spread of feature values.
     - `skewness`: Asymmetry of distribution.
     - `kurtosis`: Peak and tail heaviness.

3. **Load Pre-trained Neural Network**  
   - A **FeatureDependencyNN model** trained on linguistic data.
   - The model **predicts inter-feature relationships**.

4. **Optimize Maximum Entropy Model with NN-Based Regularization**  
   - **Finds optimal Lagrange multipliers** (`λ`) while integrating **NN predictions**.
   - Regularizes constraints **based on feature dependencies**.

5. **Store & Visualize Regularized Lagrange Multipliers**  
   - Saves **optimized λ values**.
   - Generates plots to analyze **constraint strength across linguistic features**.

---

## **📌 Methodology**
### **Step 1: Load Dataset**
- Reads **linguistic constraint features** from `fake_data_nextLevel_normalized.csv`.
- **Filters out non-numeric columns** (`Label`, `Text`).
- Converts the dataset to a **numpy array** for fast computation.

### **Step 2: Compute Constraints**
- Supports **four constraint types**:
  - **Mean** (`mean`): Average feature value.
  - **Variance** (`variance`): Spread of values.
  - **Skewness** (`skewness`): Asymmetry of distribution.
  - **Kurtosis** (`kurtosis`): Peak and tail heaviness.

### **Step 3: Load Pre-trained Neural Network**
- Loads a **feature dependency model (`FeatureDependencyNN`)**.
- The model learns **relationships between linguistic features**.
- Provides **nonlinear regularization** in entropy optimization.

### **Step 4: Define Maximum Entropy Loss Function**
The **loss function** includes:
\[
L(λ) = \sum (E_p[f] - C)^2 + α ||λ||^2 + β \sum_{i,j} \text{NN}(f_i, f_j) (λ_i - λ_j)^2
\]
where:
- **\( E_p[f] \)** = Expected constraint values under learned distribution.
- **\( C \)** = Target constraints from real data.
- **\( α ||λ||^2 \)** = L2 regularization.
- **\( β \sum \text{NN}(f_i, f_j) (λ_i - λ_j)^2 \)** = NN-based feature dependency penalty.

### **Step 5: Optimize Lagrange Multipliers**
- Uses **L-BFGS-B** optimization:
  - **Finds λ values** that minimize constraint deviation.
  - **Applies feature dependency regularization**.

### **Step 6: Store & Visualize Regularized Constraints**
- Saves **trained λ values**.
- Generates plots to compare **constraint strength across features**.

---

## **📌 Main Parameters & Their Meaning**
| Parameter | Description |
|-----------|------------|
| `data_filename` | Dataset file (`fake_data_nextLevel_normalized.csv`). |
| `CONSTRAINT_TYPE` | Type of constraint (`mean`, `variance`, `skewness`, `kurtosis`). |
| `USE_REGULARIZATION` | If `True`, applies L2 and NN-based regularization. |
| `REGULARIZATION_ALPHA` | Strength of L2 regularization (default `0.01`). |
| `NONLINEAR_DEPENDENCY_BETA` | Strength of **NN-based regularization** (default `0.05`). |
| `OPTIMIZATION_METHOD` | Optimization algorithm (`L-BFGS-B`, `BFGS`, `SLSQP`). |

---

## **📌 Outputs & File Structure**
| Output File | Description |
|------------|-------------|
| `Results/NN_MaxEnt/Lagrange_Multipliers_NN.csv` | Optimized Lagrange multipliers. |
| `Results/NN_MaxEnt/Lagrange_Multipliers_Boxplot.png` | Boxplot of constraints per linguistic feature. |
| `Results/NN_MaxEnt/Summary_Statistics_NN.csv` | Descriptive statistics of constraints. |

---

## **📌 How to Interpret the Results**
### **1️⃣ Boxplot of Lagrange Multipliers**
| Observation | Interpretation |
|-------------|---------------|
| **High λ values** | Feature is **strongly constrained**. |
| **Low λ values** | Feature is **weakly constrained**. |
| **Outliers** | Unusual constraints in data. |

### **2️⃣ Neural Network Regularization Effect**
| Result | Interpretation |
|--------|---------------|
| **Higher β value** | NN has a **stronger influence** on λ values. |
| **Lower β value** | NN influence is **weaker**, λ values are mostly unconstrained. |

---

## **📌 Limitations**
1. **Relies on a Pre-Trained Neural Network**  
   - Model performance **depends on NN quality**.
   - Future work: **Improve NN training for feature dependencies**.

2. **Fixed Regularization Strength**  
   - Uses **fixed α and β** for L2 and NN-based penalties.
   - Future work: **Optimize α and β via grid search**.

3. **Optimization Stability**  
   - **L-BFGS-B** may converge to local minima.
   - Future work: **Test alternative solvers (e.g., SLSQP, Adam).**

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

# Convert DataFrame to numpy
features = df.values

# ---- Step 2: Load Pre-trained Neural Network ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

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

# Load Model
model_path = "Models/dependency_nn_best.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Error: The trained NN model '{model_path}' was not found. "
                            "Please run 'train_dependency_nn.py' first.")
dependency_model = FeatureDependencyNN().to(device)
dependency_model.load_state_dict(torch.load(model_path, weights_only=True))
dependency_model.eval()
print(f"✅ Loaded trained model from {model_path}")

# ---- Step 3: Define User Choices ----
USE_REGULARIZATION = True
REGULARIZATION_ALPHA = 0.01  # L2 Regularization factor
NONLINEAR_DEPENDENCY_BETA = 0.05  # Weight for NN-based dependency regularization

OPTIMIZATION_METHOD = "L-BFGS-B"

CONSTRAINT_TYPE = "mean"  # Options: "mean", "variance", "skewness", "kurtosis"

# ---- Step 4: Define Constraint Function ----
def compute_constraints(df, constraint_type="mean"):
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
        raise ValueError("Invalid constraint type selected!")

constraints = compute_constraints(df, CONSTRAINT_TYPE)

# ---- Step 5: Define Maximum Entropy Loss Function with NN-Based Regularization ----
def max_entropy_loss(lambda_vals, features, constraints, alpha=0.01, beta=0.05, nn_model=None):
    exp_term = np.exp(-np.dot(features, lambda_vals))
    Z = np.sum(exp_term)
    p_x = exp_term / Z
    expected_constraints = np.dot(p_x, features)
    
    loss = np.sum((expected_constraints - constraints) ** 2)
    
    if USE_REGULARIZATION:
        loss += alpha * np.sum(lambda_vals ** 2)

        # NN Regularization: Encourage similarity in lambda values based on NN-predicted dependencies
        nn_dependency_penalty = 0
        for i in range(len(lambda_vals)):
            for j in range(i + 1, len(lambda_vals)):  
                nn_input = torch.tensor([features[:, i].mean(), features[:, j].mean()], dtype=torch.float32).unsqueeze(0).to(device)
                nn_prediction = nn_model(nn_input).item()
                nn_dependency_penalty += nn_prediction * (lambda_vals[i] - lambda_vals[j]) ** 2

        loss += beta * nn_dependency_penalty

    return loss

# ---- Step 6: Optimize Lagrange Multipliers ----
lambda_init = np.random.uniform(-0.1, 0.1, df.shape[1])

opt_result = opt.minimize(
    max_entropy_loss,
    lambda_init,
    args=(features, constraints, REGULARIZATION_ALPHA, NONLINEAR_DEPENDENCY_BETA, dependency_model),
    method=OPTIMIZATION_METHOD
)

lambda_values = opt_result.x
print("✅ Optimization completed using NN-based feature dependencies.")

# ---- Step 7: Organize and Save Results ----
results_dir = "Results/NN_MaxEnt"
os.makedirs(results_dir, exist_ok=True)

lambda_df = pd.DataFrame({"Index": df.columns, "Lambda": lambda_values})

# Save Lagrange Multipliers
lambda_df.to_csv(os.path.join(results_dir, "Lagrange_Multipliers_NN.csv"), index=False)

# ---- Step 8: Visualization ----
plt.figure(figsize=(10, 6))
sns.boxplot(x=lambda_df["Index"], y=lambda_df["Lambda"])
plt.xticks(rotation=45, ha="right")
plt.title("NN-Based Regularized Lagrange Multipliers")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Lagrange_Multipliers_Boxplot.png"), dpi=300)
plt.savefig(os.path.join(results_dir, "Lagrange_Multipliers_Boxplot.pdf"), dpi=300)
plt.show()

# Save Summary Statistics
summary_stats = lambda_df.describe()
summary_stats.to_csv(os.path.join(results_dir, "Summary_Statistics_NN.csv"))

# Print Summary Statistics
print(summary_stats)

print(f"✅ All results and plots have been saved in {results_dir}/")
