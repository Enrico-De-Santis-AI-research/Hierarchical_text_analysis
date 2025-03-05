import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.optimize as opt
import os
import seaborn as sns
import matplotlib.pyplot as plt

"""
# 📊 **Maximum Entropy Training with Neural Network Dependency Regularization**
### **Author:** Enrico De Santis
### **Version:** 1.0  
### **Last Updated:** [Date]  
### **Dependencies:** NumPy, Pandas, Torch (PyTorch), SciPy, Seaborn, Matplotlib  

---

## **🔍 Overview**
This module **trains a Maximum Entropy Model** with dependency regularization using a **pre-trained neural network** to account for feature dependencies. The goal is to estimate **Lagrange multipliers (λ values)** for different linguistic constraints (e.g., morphology, syntax, semantics) while **enforcing structured dependencies** between them.

---

## **🎯 Objectives**
1. **Load Preprocessed Training and Test Datasets**  
   - Extract only **numerical constraint features** from `fake_data_nextLevel_normalized.csv` (training) and `fake_data_test_normalized.csv` (test).  
   - Ensure correct data types (float32) for compatibility.

2. **Retrieve Optimal Hyperparameters from Previous Grid Search**  
   - Loads the best **alpha** (L2 regularization) and **beta** (dependency penalty) values.  
   - Ensures that training is performed with the best hyperparameter combination.

3. **Load a Pre-trained Neural Network to Model Feature Dependencies**  
   - The **FeatureDependencyNN** predicts dependency strength between pairs of features.  
   - These dependencies introduce **non-linear regularization terms** into the Maximum Entropy objective function.

4. **Train the Maximum Entropy Model Using L-BFGS-B Optimization**  
   - The optimization function minimizes the difference between **expected and empirical constraints**.  
   - **Neural network dependency regularization** prevents violations of natural linguistic hierarchies.

5. **Evaluate Generalization on Unseen Test Data**  
   - Compute **test loss** to assess the model’s performance on new samples.

6. **Save & Visualize Lagrange Multipliers**  
   - Store trained Lagrange multipliers for further interpretation.  
   - Generate **bar plots** showing constraint strengths across different linguistic levels.

---

## **📌 Methodology**
### **Step 1: Load Dataset**
- Loads **training** and **test** data from the `Data/` folder.
- Filters out **non-numeric columns** (`Label`, `Text`) to keep only feature indices.

### **Step 2: Compute Constraints for Maximum Entropy**
- Supports different types of constraints:
  - **Mean:** Average value of each feature.
  - **Variance:** Measures variability.
  - **Skewness:** Asymmetry of distribution.
  - **Kurtosis:** Tail heaviness.

### **Step 3: Load Best Hyperparameters**
- Reads **best α (L2 regularization) and β (dependency penalty)** from previous `grid_search_maxent.py` results.

### **Step 4: Load Pre-trained Neural Network for Feature Dependency Regularization**
- **FeatureDependencyNN:** A **small feedforward neural network** trained to model **pairwise dependencies** between linguistic features.
- Used to **regularize the Maximum Entropy model**, ensuring **linguistic hierarchy preservation**.

### **Step 5: Define Maximum Entropy Loss Function**
The optimization objective is:

\[
\mathcal{L} = \sum (E_{\text{model}}[f] - E_{\text{data}}[f])^2 + \alpha ||\lambda||^2 + \beta \sum_{\text{pairs}} \text{NN}(x_i, x_j) \cdot (\lambda_i - \lambda_j)^2
\]

Where:
- **First term:** Minimizes constraint violations.
- **Second term (α L2 regularization):** Prevents extreme λ values.
- **Third term (β NN regularization):** Penalizes λ values that violate learned feature dependencies.

### **Step 6: Train the Maximum Entropy Model**
- Uses **L-BFGS-B** to find optimal **Lagrange multipliers (λ values)**.
- The optimizer **minimizes constraint deviations** while respecting **linguistic hierarchy dependencies**.

### **Step 7: Evaluate on Test Set**
- Computes **test loss** using the trained λ values.
- If test loss is **high**, the model may **overfit** the training data.

### **Step 8: Save & Visualize Results**
- Saves **trained λ values** in `Results/Trained_MaxEnt_[constraint_type]/`.
- Generates a **barplot** of **constraint strength per linguistic level**.

---

## **📌 Main Parameters & Their Meaning**
| Parameter | Description |
|-----------|------------|
| `train_data_filename` | Training dataset file (`fake_data_nextLevel_normalized.csv`). |
| `test_data_filename` | Test dataset file (`fake_data_test_normalized.csv`). |
| `CONSTRAINT_TYPE` | Defines how constraints are computed (`mean`, `variance`, `skewness`, `kurtosis`). |
| `best_alpha` | L2 regularization strength (from grid search). |
| `best_beta` | Neural dependency regularization strength (from grid search). |
| `model_path` | Path to pre-trained **FeatureDependencyNN** model. |
| `lambda_init` | Initial Lagrange multipliers (randomized between -0.1 and 0.1). |

---

## **📌 Outputs & File Structure**
| Output File | Description |
|------------|-------------|
| `Results/Trained_MaxEnt_[constraint_type]/Trained_Lagrange_Multipliers.csv` | Stores trained λ values. |
| `Results/Trained_MaxEnt_[constraint_type]/Lagrange_Multipliers.png` | Barplot of constraint strengths per level. |

---

## **📌 How to Interpret the Results**
### **1️⃣ Test Loss Analysis**
| Test Loss | Interpretation |
|-----------|---------------|
| **< 0.05** | ✅ **Excellent generalization** (model performs well on unseen data). |
| **0.05 - 0.2** | ⚠️ **Moderate generalization** (some risk of overfitting, needs validation). |
| **> 0.2** | ❌ **High overfitting risk** (model may not generalize well). |

### **2️⃣ Constraint Strength Across Levels**
- **If lower levels (character, morphology, syntax) have higher λ values:**  
  ✅ *Supports the hypothesis that lower levels impose stronger constraints.*  
- **If higher levels (semantics, stylistics) have unexpectedly strong constraints:**  
  ❌ *Suggests that conceptual constraints may have a greater impact than expected.*

### **3️⃣ Visualizations**
- **Barplot:** Highlights constraint strengths per linguistic level.

---

## **📌 Limitations**
1. **Dependency Model Assumes Pairwise Interactions**  
   - **Does not model higher-order interactions** (e.g., triplets of features).

2. **Test Loss Interpretation May Be Simplistic**  
   - A **low test loss** does not always indicate a well-generalized model.
   - **Cross-validation or held-out validation sets** may improve robustness.

3. **Effect of Different Constraint Types Not Fully Explored**  
   - Currently supports **mean, variance, skewness, kurtosis**.
   - Exploring **entropy-based constraints** could yield deeper insights.

---
"""

# ---- Step 1: Load Dataset ----
data_dir = "Data"
train_data_filename = "fake_data_nextLevel_normalized.csv"
test_data_filename = "fake_data_test_normalized.csv"

train_data_path = os.path.join(data_dir, train_data_filename)
test_data_path = os.path.join(data_dir, test_data_filename)

if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
    raise FileNotFoundError(f"❌ Missing dataset. Run 'generate_fake_data_nextLevel.py' and 'generate_fake_test_data.py' first.")

df_train = pd.read_csv(train_data_path)
df_test = pd.read_csv(test_data_path)

# ---- Step 2: Filter Only Numerical Columns ----
non_numeric_cols = ["Label", "Text"]
index_columns = [col for col in df_train.columns if col not in non_numeric_cols]

features_train = df_train[index_columns].values.astype(np.float32)  # Convert to float32
features_test = df_test[index_columns].values.astype(np.float32)

# ---- Step 3: Load Best Hyperparameters ----
CONSTRAINT_TYPE = "mean"  # Options: "mean", "variance", "skewness", "kurtosis"

def compute_constraints(df, constraint_type):
    """Compute constraint values based on the selected method."""
    if constraint_type == "mean":
        return df.mean().values.astype(np.float32)
    elif constraint_type == "variance":
        return df.var().values.astype(np.float32)
    elif constraint_type == "skewness":
        return df.skew().values.astype(np.float32)
    elif constraint_type == "kurtosis":
        return df.kurtosis().values.astype(np.float32)
    else:
        raise ValueError(f"❌ Error: Invalid constraint type '{constraint_type}' selected!")

best_params_path = f"Results/Grid_Search_{CONSTRAINT_TYPE}/Grid_Search_Results.csv"
if not os.path.exists(best_params_path):
    raise FileNotFoundError(f"❌ Grid search results not found in {best_params_path}. Run 'grid_search_maxent.py' first.")

grid_results = pd.read_csv(best_params_path)
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
dependency_model = FeatureDependencyNN()
dependency_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))  # Use weights_only=True
dependency_model.eval()

# ---- Step 5: Define Maximum Entropy Loss ----
def max_entropy_loss(lambda_vals, features, constraints, alpha, beta, nn_model):
    exp_term = np.exp(-np.dot(features, lambda_vals))
    Z = np.sum(exp_term)
    p_x = exp_term / Z
    expected_constraints = np.dot(p_x, features)
    
    loss = np.sum((expected_constraints - constraints) ** 2)
    loss += alpha * np.sum(lambda_vals ** 2)

    nn_dependency_penalty = sum(
        (nn_model(torch.tensor([features[:, i].mean(), features[:, j].mean()], dtype=torch.float32).unsqueeze(0)).item()
         * (lambda_vals[i] - lambda_vals[j]) ** 2)
        for i in range(len(lambda_vals)) for j in range(i + 1, len(lambda_vals))
    )
    
    loss += beta * nn_dependency_penalty
    return loss

# ---- Step 6: Train on Training Set ----
train_constraints = compute_constraints(df_train[index_columns], CONSTRAINT_TYPE)  # Convert to float32
lambda_init = np.random.uniform(-0.1, 0.1, len(index_columns)).astype(np.float32)  # Ensure consistency

opt_result = opt.minimize(
    max_entropy_loss,
    lambda_init,
    args=(features_train, train_constraints, best_alpha, best_beta, dependency_model),
    method="L-BFGS-B"
)

lambda_values = opt_result.x

# ---- Step 7: Evaluate on Test Set ----
test_constraints = compute_constraints(df_test[index_columns], CONSTRAINT_TYPE)  # Convert to float32

test_loss = max_entropy_loss(lambda_values, features_test, test_constraints, best_alpha, best_beta, dependency_model)
print(f"✅ Test Loss: {test_loss:.6f}")

# ---- Step 8: Save & Plot Results ----
results_dir = f"Results/Trained_MaxEnt_{CONSTRAINT_TYPE}"
os.makedirs(results_dir, exist_ok=True)

lambda_df = pd.DataFrame({"Index": index_columns, "Lambda": lambda_values})
lambda_df.to_csv(os.path.join(results_dir, "Trained_Lagrange_Multipliers.csv"), index=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=lambda_df["Index"], y=lambda_df["Lambda"], color='royalblue')
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.title(f"Lagrange Multipliers (Trained on Best Hyperparameters) - {CONSTRAINT_TYPE.capitalize()}")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Lagrange_Multipliers.png"), dpi=300)
plt.show()

print(f"✅ Results saved in {results_dir}/")
