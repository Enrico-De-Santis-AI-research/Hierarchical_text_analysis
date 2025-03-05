import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
import os

"""
# 📊 **Regularized Maximum Entropy Model for Linguistic Constraints**
### **Author:** Enrico De Santis 
### **Version:** 1.2  
### **Last Updated:** [Date]  
### **Dependencies:** NumPy, Pandas, SciPy, Seaborn, Matplotlib, OS

---

## **🔍 Overview**
This script implements a **Maximum Entropy Model (MaxEnt)** for analyzing linguistic constraints  
across **hierarchical levels of text structure** (e.g., character, morphology, syntax, semantics).  

The **regularization term** allows for **smoother constraint estimation** and prevents overfitting.  

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

3. **Optimize a Maximum Entropy Model**  
   - **Finds optimal Lagrange multipliers** (`λ`) to satisfy constraints.
   - **Uses regularization** (L2 norm) to stabilize learning.

4. **Interpret Constraints Hierarchically**  
   - Groups constraints into **linguistic levels**:
     - **Character Level** (e.g., Zipf’s law, complexity).
     - **Morphology Level** (e.g., syllables, lexical diversity).
     - **Syntax Level** (e.g., POS distribution, sentence length).
     - **Grammar Level** (e.g., readability metrics).
     - **Semantic Level** (e.g., similarity measures, topic modeling).
     - **Stylistic Level** (e.g., BLEU, ROUGE, text flow).

5. **Visualize & Validate Hypothesis**  
   - Generates plots to compare constraint strength across levels.

---

## **📌 Methodology**
### **Step 1: Load Dataset**
- Reads **linguistic constraint features** from `fake_data_nextLevel_normalized.csv`.
- **Filters out non-numeric columns** (`Label`, `Text`).
- Converts the dataset to a **numpy array** for faster computation.

### **Step 2: Compute Constraints**
- Supports **four constraint types**:
  - **Mean** (`mean`): Average feature value.
  - **Variance** (`variance`): Spread of values.
  - **Skewness** (`skewness`): Asymmetry of distribution.
  - **Kurtosis** (`kurtosis`): Peak and tail heaviness.

### **Step 3: Define Maximum Entropy Loss Function**
- Defines the **loss function** as:
  \[
  L(λ) = \sum (E_p[f] - C)^2 + α ||λ||^2
  \]
  where:
  - **\( E_p[f] \)** = Expected constraint values under learned distribution.
  - **\( C \)** = Target constraints from real data.
  - **\( α ||λ||^2 \)** = L2 regularization.

### **Step 4: Optimize Lagrange Multipliers**
- Uses **L-BFGS-B** optimization:
  - **Finds λ values** that minimize constraint deviation.
  - **Applies regularization** to prevent overfitting.

### **Step 5: Store & Structure Constraints Hierarchically**
- Groups constraints into **linguistic levels**:
  - **Lower levels (Character, Morphology, Syntax)** impose **stronger constraints**.
  - **Higher levels (Semantics, Style)** have **more flexibility**.

### **Step 6: Generate & Save Plots**
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
| `REGULARIZATION_ALPHA` | Regularization strength (default `0.01`). |
| `OPTIMIZATION_METHOD` | Optimization algorithm (`L-BFGS-B`, `BFGS`, `SLSQP`). |

---

## **📌 Outputs & File Structure**
| Output File | Description |
|------------|-------------|
| `Results/Regularized_MaxEnt_independent/Lagrange_Multipliers.csv` | Lagrange multipliers per feature. |
| `Results/Regularized_MaxEnt_independent/Lagrange_Multipliers_Boxplot.png` | Boxplot of constraints per level. |
| `Results/Regularized_MaxEnt_independent/Average_Constraint_Strength.png` | Barplot of constraint strength. |
| `Results/Regularized_MaxEnt_independent/Summary_Statistics.csv` | Descriptive statistics of constraints. |

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
1. **Assumes Feature Independence**  
   - Each constraint is **optimized separately**.
   - Future improvements: **add feature dependency modeling**.

2. **Regularization Strength Fixed**  
   - Uses a **fixed α = 0.01**.
   - Future work: **tune α dynamically via grid search**.

3. **Optimization Stability**  
   - **L-BFGS-B** may converge to local minima.
   - Future work: **test alternative solvers (e.g., SLSQP, Adam).**

---
"""


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

# ---- Step 3: User Choices for Regularization, Optimization, and Constraints ----
USE_REGULARIZATION = True  # Set False to disable regularization
REGULARIZATION_ALPHA = 0.01  # L2 Regularization factor

OPTIMIZATION_METHOD = "L-BFGS-B"  # Choose between "BFGS", "L-BFGS-B", "SLSQP"

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

# ---- Step 5: Define Maximum Entropy Loss Function ----
def max_entropy_loss(lambda_vals, features, constraints, alpha=0.01, regularization=True):
    """Objective function for Maximum Entropy: Finds the best Lagrange multipliers."""
    exp_term = np.exp(-np.dot(features, lambda_vals))
    Z = np.sum(exp_term)
    p_x = exp_term / Z
    expected_constraints = np.dot(p_x, features)
    
    loss = np.sum((expected_constraints - constraints) ** 2)  # Minimize deviation
    
    if regularization:
        loss += alpha * np.sum(lambda_vals ** 2)  # L2 regularization

    return loss

# ---- Step 6: Optimization ----
lambda_init = np.random.uniform(-0.1, 0.1, df.shape[1])  # Better initialization

opt_result = opt.minimize(
    max_entropy_loss,
    lambda_init,
    args=(df.values, constraints, REGULARIZATION_ALPHA, USE_REGULARIZATION),
    method=OPTIMIZATION_METHOD
)

lambda_values = opt_result.x

# ---- Step 7: Store and Structure Lagrange Multipliers in a Hierarchical Form ----
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

# ---- Step 8: Save Results and Generate Plots ----
results_dir = "Results"
experiment_name = "Regularized_MaxEnt_independent"  # Change this name for different experiments
experiment_results_dir = os.path.join(results_dir, experiment_name)
os.makedirs(experiment_results_dir, exist_ok=True)


# Save Lagrange Multipliers
lambda_df.to_csv(os.path.join(experiment_results_dir, "Lagrange_Multipliers.csv"), index=False)

# Boxplot of Lagrange Multipliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=lambda_df["Level"], y=lambda_df["Lambda"])
plt.xticks(rotation=45, ha="right")
plt.title("Lagrange Multipliers per Linguistic Hierarchy Level")
plt.tight_layout()
plt.savefig(os.path.join(experiment_results_dir, "Lagrange_Multipliers_Boxplot.png"), dpi=300)
plt.savefig(os.path.join(experiment_results_dir, "Lagrange_Multipliers_Boxplot.pdf"), dpi=300)
plt.show()

# Check if lower-level constraints have higher lambda values (stronger constraints)
level_means = lambda_df.groupby("Level")["Lambda"].mean()
sorted_levels = ["Character_Level", "Morphology_Level", "Syntax_Level", "Grammar_Level", "Semantic_Level", "Stylistic_Level"]

# Barplot of Average Constraint Strength
plt.figure(figsize=(12, 6))
plt.bar(sorted_levels, [level_means[l] for l in sorted_levels], color='royalblue')
plt.xticks(rotation=30, ha="right", fontsize=12)
plt.title("Average Constraint Strength (Lagrange Multipliers) per Hierarchical Level", fontsize=14)
plt.ylabel("Average Lambda Value", fontsize=12)
plt.xlabel("Linguistic Hierarchy Level", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(rexperiment_results_dir, "Average_Constraint_Strength.png"), dpi=300)
plt.savefig(os.path.join(experiment_results_dir, "Average_Constraint_Strength.pdf"), dpi=300)
plt.show()

# Save Summary Statistics
summary_stats = lambda_df.groupby("Level")["Lambda"].describe()
summary_stats.to_csv(os.path.join(experiment_results_dir, "Summary_Statistics.csv"))

# Print summary statistics
print(summary_stats)

# Print Hypothesis Test Conclusion
if level_means["Character_Level"] > level_means["Semantic_Level"]:
    print("✅ Hypothesis Supported: Lower levels (Character, Morphology, Syntax) impose stronger constraints than higher levels.")
else:
    print("❌ Hypothesis Rejected: Higher levels impose stronger constraints, or results are inconclusive.")

print(f"✅ All results and plots have been saved in the '{experiment_results_dir}' folder.")
