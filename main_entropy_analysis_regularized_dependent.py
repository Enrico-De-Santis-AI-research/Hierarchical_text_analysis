import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
import os

"""
# 📊 **Regularized Maximum Entropy Model with Feature Dependency**
### **Author:** Enrico De Santis
### **Version:** 1.3  
### **Last Updated:** [Date]  
### **Dependencies:** NumPy, Pandas, SciPy, Seaborn, Matplotlib, OS  

---

## **🔍 Overview**
This script extends the **Maximum Entropy Model (MaxEnt)** by incorporating **feature dependency regularization**  
through **correlation-based penalties**. The goal is to **constrain linguistic features** within hierarchical levels  
while accounting for feature relationships.

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

3. **Calculate Feature Correlations**  
   - Computes **Pearson correlation** between linguistic features.
   - Applies **correlation-based regularization**.

4. **Optimize Maximum Entropy Model with Feature Dependencies**  
   - **Finds optimal Lagrange multipliers** (`λ`) while incorporating **correlation penalties**.
   - Uses **L2 regularization** to smooth λ values.

5. **Interpret Constraints in a Hierarchical Form**  
   - Groups constraints into **linguistic levels**:
     - **Character Level** (e.g., Zipf’s law, complexity).
     - **Morphology Level** (e.g., syllables, lexical diversity).
     - **Syntax Level** (e.g., POS distribution, sentence length).
     - **Grammar Level** (e.g., readability metrics).
     - **Semantic Level** (e.g., similarity measures, topic modeling).
     - **Stylistic Level** (e.g., BLEU, ROUGE, text flow).

6. **Visualize & Validate Hierarchical Hypothesis**  
   - Generates plots to compare **constraint strength across levels**.

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

### **Step 3: Compute Feature Correlations**
- Computes **Pearson correlation matrix** between features.
- Uses correlation values to **apply dependency penalties** in optimization.

### **Step 4: Define Maximum Entropy Loss Function**
- The **loss function** includes:
  \[
  L(λ) = \sum (E_p[f] - C)^2 + α ||λ||^2 + β \sum_{i,j} r_{ij} (λ_i - λ_j)^2
  \]
  where:
  - **\( E_p[f] \)** = Expected constraint values under learned distribution.
  - **\( C \)** = Target constraints from real data.
  - **\( α ||λ||^2 \)** = L2 regularization.
  - **\( β \sum r_{ij} (λ_i - λ_j)^2 \)** = Feature dependency penalty (correlation matrix \( r_{ij} \)).

### **Step 5: Optimize Lagrange Multipliers**
- Uses **L-BFGS-B** optimization:
  - **Finds λ values** that minimize constraint deviation.
  - **Applies feature dependency regularization**.

### **Step 6: Store & Structure Constraints Hierarchically**
- Groups constraints into **linguistic levels**:
  - **Lower levels (Character, Morphology, Syntax)** impose **stronger constraints**.
  - **Higher levels (Semantics, Style)** have **more flexibility**.

### **Step 7: Generate & Save Plots**
- **Boxplot of Lagrange Multipliers per Level**
  - Shows **constraint strength** at each linguistic level.
- **Barplot of Average Constraint Strength**
  - Tests **hierarchy hypothesis** (stronger constraints at lower levels).

---

## **📌 Main Parameters & Their Meaning**
| Parameter | Description |
|-----------|------------|
| `data_filename` | Dataset file (`fake_data_nextLevel_normalized.csv`). |
| `CONSTRAINT_TYPE` | Type of constraint (`mean`, `variance`, `skewness`, `kurtosis`). |
| `USE_REGULARIZATION` | If `True`, applies L2 regularization. |
| `REGULARIZATION_ALPHA` | Strength of L2 regularization (default `0.01`). |
| `CORRELATION_REGULARIZATION_BETA` | Strength of **correlation-based regularization** (default `0.05`). |
| `OPTIMIZATION_METHOD` | Optimization algorithm (`L-BFGS-B`, `BFGS`, `SLSQP`). |

---

## **📌 Outputs & File Structure**
| Output File | Description |
|------------|-------------|
| `Results/Regularized_MaxEnt/Lagrange_Multipliers.csv` | Lagrange multipliers per feature. |
| `Results/Regularized_MaxEnt/Lagrange_Multipliers_Boxplot.png` | Boxplot of constraints per level. |
| `Results/Regularized_MaxEnt/Average_Constraint_Strength.png` | Barplot of constraint strength. |
| `Results/Regularized_MaxEnt/Summary_Statistics.csv` | Descriptive statistics of constraints. |

---

## **📌 How to Interpret the Results**
### **1️⃣ Boxplot of Lagrange Multipliers**
| Observation | Interpretation |
|-------------|---------------|
| **High λ values** | Feature is **strongly constrained**. |
| **Low λ values** | Feature is **weakly constrained**. |
| **Outliers** | Unusual constraints in data. |

### **2️⃣ Barplot of Average Constraint Strength**
| Result | Interpretation |
|--------|---------------|
| **Character-Level λ > Semantic-Level λ** | ✅ **Hypothesis confirmed**: Low-level constraints dominate. |
| **Character-Level λ < Semantic-Level λ** | ❌ **Unexpected result**: Check dataset and constraints. |

---

## **📌 Limitations**
1. **Assumes Linear Feature Dependencies**  
   - Uses **correlation matrix**, which **only captures linear dependencies**.
   - Future improvement: **use neural networks to model nonlinear dependencies**.

2. **Fixed Regularization Strength**  
   - Uses **fixed α and β** for L2 and correlation-based penalties.
   - Future work: **optimize α and β via grid search**.

3. **Optimization Stability**  
   - **L-BFGS-B** may converge to local minima.
   - Future work: **test alternative solvers (e.g., SLSQP, Adam).**

---
""""

# ---- Step 1: Load Dataset ----
data_dir = "Data"

# User-defined dataset file (change this to load a different dataset)
data_filename = "fake_data_nextLevel_normalized.csv"  # Change this to load other datasets
data_path = os.path.join(data_dir, data_filename)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Error: The dataset '{data_filename}' was not found in '{data_dir}/'. "
                            "Please check the filename or run 'generate_fake_data_nextLevel.py' first.")

print(f"✅ Loading dataset: {data_filename}")
df = pd.read_csv(data_path)

# ---- Step 2: Select Only Numerical Index Columns ----
# Skip "Label" and "Text" columns
index_columns = [
    "Zipf_Law", "Kolmogorov_Complexity",
    "Syllable_Counts", "Word_Lengths", "Lexical_Diversity_TTR", "Lexical_Diversity_Shannon",
    "POS_Tag_Distribution", "Avg_Sentence_Length", "Dependency_Complexity", "Punctuation_Usage",
    "Readability_Dale_Chall", "Readability_Flesch_Kincaid", "Referential_Integrity",
    "Cosine_Similarity", "WordNet_Similarity", "Lexical_Chains", "LDA_Topic_Consistency",
    "Style_Consistency", "BLEU_Score", "Jaccard_Score", "ROUGE_Score", "Text_Flow_Analysis"
]

df = df[index_columns]  # Keep only numerical indices

# ---- Step 3: Compute Feature Correlations ----
correlation_matrix = df.corr()  # Pearson correlation between features
correlation_matrix.fillna(0, inplace=True)  # Handle NaN cases if any

# ---- Step 4: User Choices for Regularization, Optimization, and Constraints ----
USE_REGULARIZATION = True  # Set False to disable correlation regularization
REGULARIZATION_ALPHA = 0.01  # L2 Regularization factor on individual lambda values
CORRELATION_REGULARIZATION_BETA = 0.05  # New regularization on correlated lambda values

OPTIMIZATION_METHOD = "L-BFGS-B"  # Choose between "BFGS", "L-BFGS-B", "SLSQP"

CONSTRAINT_TYPE = "mean"  # Options: "mean", "variance", "skewness", "kurtosis"

# ---- Step 5: Define Constraint Function ----
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

# ---- Step 6: Define Maximum Entropy Loss Function with Feature Dependency Regularization ----
def max_entropy_loss(lambda_vals, features, constraints, alpha=0.01, beta=0.05, correlation_matrix=None, regularization=True):
    """Objective function for Maximum Entropy: Finds the best Lagrange multipliers with feature dependency regularization."""
    exp_term = np.exp(-np.dot(features, lambda_vals))
    Z = np.sum(exp_term)
    p_x = exp_term / Z
    expected_constraints = np.dot(p_x, features)
    
    loss = np.sum((expected_constraints - constraints) ** 2)  # Minimize deviation
    
    if regularization:
        # L2 Regularization (ensures smooth lambda values)
        loss += alpha * np.sum(lambda_vals ** 2)
        
        # Correlation Regularization (penalize large lambda differences for correlated features)
        correlation_penalty = 0
        for i in range(len(lambda_vals)):
            for j in range(i + 1, len(lambda_vals)):  # Avoid duplicate calculations
                correlation_penalty += correlation_matrix.iloc[i, j] * (lambda_vals[i] - lambda_vals[j]) ** 2
        
        loss += beta * correlation_penalty  # Weight correlation-based regularization
    
    return loss

# ---- Step 7: Optimization ----
lambda_init = np.random.uniform(-0.1, 0.1, df.shape[1])  # Better initialization

opt_result = opt.minimize(
    max_entropy_loss,
    lambda_init,
    args=(df.values, constraints, REGULARIZATION_ALPHA, CORRELATION_REGULARIZATION_BETA, correlation_matrix, USE_REGULARIZATION),
    method=OPTIMIZATION_METHOD
)

lambda_values = opt_result.x

# ---- Step 8: Store and Structure Lagrange Multipliers in a Hierarchical Form ----
hierarchy = {
    "Character_Level": ["Zipf_Law", "Kolmogorov_Complexity"],
    "Morphology_Level": ["Syllable_Counts", "Word_Lengths", "Lexical_Diversity_TTR", "Lexical_Diversity_Shannon"],
    "Syntax_Level": ["POS_Tag_Distribution", "Avg_Sentence_Length", "Dependency_Complexity", "Punctuation_Usage"],
    "Grammar_Level": ["Readability_Dale_Chall", "Readability_Flesch_Kincaid", "Referential_Integrity"],
    "Semantic_Level": ["Cosine_Similarity", "WordNet_Similarity", "Lexical_Chains", "LDA_Topic_Consistency"],
    "Stylistic_Level": ["Style_Consistency", "BLEU_Score", "Jaccard_Score", "ROUGE_Score", "Text_Flow_Analysis"]
}

lambda_df = pd.DataFrame({"Index": df.columns, "Lambda": lambda_values})
lambda_df["Level"] = lambda_df["Index"].map(
    {idx: level for level, indices in hierarchy.items() for idx in indices}
)

# ---- Step 9: Create Subfolder for Results ----
results_dir = "Results"
experiment_name = "Regularized_MaxEnt"  # Change this name for different experiments
experiment_results_dir = os.path.join(results_dir, experiment_name)
os.makedirs(experiment_results_dir, exist_ok=True)

# ---- Step 10: Save Results and Generate Plots ----
# Save Lagrange Multipliers
lambda_df.to_csv(os.path.join(experiment_results_dir, "Lagrange_Multipliers.csv"), index=False)

# Boxplot of Lagrange Multipliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=lambda_df["Level"], y=lambda_df["Lambda"])
plt.xticks(rotation=45, ha="right")
plt.title("Regularized Lagrange Multipliers per Linguistic Hierarchy Level")
plt.tight_layout()
plt.savefig(os.path.join(experiment_results_dir, "Lagrange_Multipliers_Boxplot.png"), dpi=300)
plt.savefig(os.path.join(experiment_results_dir, "Lagrange_Multipliers_Boxplot.pdf"), dpi=300)
plt.show()

# Save Summary Statistics
summary_stats = lambda_df.groupby("Level")["Lambda"].describe()
summary_stats.to_csv(os.path.join(experiment_results_dir, "Summary_Statistics.csv"))

# Print summary statistics
print(summary_stats)

print(f"✅ All results and plots have been saved in the '{experiment_results_dir}' folder.")
