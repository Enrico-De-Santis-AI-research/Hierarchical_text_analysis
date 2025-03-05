import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


"""
# 📊 **Maximum Entropy Model Interpretation**
### **Author:** Enrico De Santis 
### **Version:** 1.0  
### **Last Updated:** [Date]  
### **Dependencies:** NumPy, Pandas, Seaborn, Matplotlib  

## **🔍 Overview**
This module interprets the **Maximum Entropy Model's Lagrange multipliers**, which quantify the constraints imposed at different linguistic levels (character, morphology, syntax, grammar, semantics, stylistics). It also evaluates the model's **generalization to unseen data** using the test loss.

## **🎯 Objectives**
1. **Summarize Constraint Strength per Linguistic Level**  
   - Compute and compare **mean constraint strength** at different levels.
   - Test the hypothesis: *Lower levels (character, morphology) impose stronger constraints than higher levels (semantics, stylistics).*

2. **Evaluate Model Generalization**  
   - Load and interpret **test loss** to assess overfitting/generalization.
   - Provide **warnings** if the test loss suggests poor generalization.

3. **Generate a Comprehensive Interpretation Report**  
   - Summarize **constraint strengths** and **hypothesis testing**.
   - Store results in `Results/Interpretation_Report.txt`.

4. **Visualize the Findings**  
   - **Boxplot** of Lagrange multipliers per level.
   - **Barplot** of mean constraint strength across linguistic levels.

---

## **📌 Methodology**
1. **Load the Latest Trained Model Results**  
   - Automatically selects the most recent model trained with constraints (`mean`, `variance`, `skewness`, or `kurtosis`).  
   - Reads **Lagrange multipliers** from `Trained_Lagrange_Multipliers.csv`.  
   - If available, loads **test loss** from `Test_Loss.txt`.

2. **Group Constraints by Linguistic Level**  
   - **Character-Level:** Zipf’s Law, Kolmogorov Complexity  
   - **Morphology-Level:** Syllables, Word Length, Lexical Diversity  
   - **Syntax-Level:** POS Tags, Sentence Length, Punctuation  
   - **Grammar-Level:** Readability Scores, Referential Integrity  
   - **Semantic-Level:** Cosine Similarity, WordNet Similarity  
   - **Stylistic-Level:** Style Consistency, BLEU, ROUGE, Jaccard  

3. **Compute Constraint Strength per Level**  
   - Computes **mean lambda values** per linguistic level.
   - Checks whether **lower-level constraints** are stronger than higher levels.

4. **Generate Interpretation Report**  
   - Stores constraint strengths and hypothesis evaluation in `Interpretation_Report.txt`.  
   - **Interprets test loss**:
     - **Low test loss (< 0.05)** → Model generalizes well ✅  
     - **Moderate test loss (0.05 - 0.2)** → Some generalization issues ⚠️  
     - **High test loss (> 0.2)** → Possible overfitting ❌  

5. **Generate & Save Plots**  
   - **Boxplot** of Lagrange multipliers per level.  
   - **Barplot** of average constraint strength per level.  

---

## **📌 Main Parameters & Their Meaning**
| Parameter | Description |
|-----------|------------|
| `results_dir` | Directory where trained models are stored (`Results/`). |
| `constraint_types` | List of constraints applied (`mean`, `variance`, `skewness`, `kurtosis`). |
| `trained_dir` | Path to the latest trained model directory. |
| `lambda_file` | CSV file containing estimated Lagrange multipliers. |
| `test_loss_file` | File storing test loss, used to assess generalization. |

---

## **📌 Outputs & File Structure**
| Output File | Description |
|------------|-------------|
| `Results/Interpretation_Report.txt` | Summary of constraint strengths, hypothesis testing, and test loss analysis. |
| `Results/Trained_MaxEnt_[constraint]/Interpretation_Plots/` | Folder containing plots. |
| `Lagrange_Multipliers_Boxplot.png` | Boxplot showing variation in constraints per level. |
| `Average_Constraint_Strength.png` | Barplot of average constraint strength per level. |

---

## **📌 How to Interpret the Results**
### **1️⃣ Test Loss Analysis**
| Test Loss | Interpretation |
|-----------|---------------|
| **< 0.05** | ✅ **Excellent generalization** (model performs well on unseen data). |
| **0.05 - 0.2** | ⚠️ **Moderate generalization** (some risk of overfitting, needs validation). |
| **> 0.2** | ❌ **High overfitting risk** (model may not generalize well). |

### **2️⃣ Constraint Strength Across Levels**
- **If lower levels (character, morphology, syntax) have higher lambda values:**  
  ✅ *Supports the hypothesis that lower levels impose stronger constraints.*  
- **If higher levels (semantics, stylistics) have unexpectedly strong constraints:**  
  ❌ *Suggests that conceptual constraints may have a greater impact than expected.*

### **3️⃣ Visualizations**
- **Boxplot:** Checks distribution of constraints across levels.  
- **Barplot:** Identifies the levels with the strongest constraints.

---

## **📌 Limitations**
1. **Test Loss Interpretation May Be Simplistic**  
   - A **low test loss** does not always mean a perfect model.
   - More robust validation (cross-validation) should be considered.

2. **Assumption of Hierarchical Constraint Strength**  
   - The hypothesis assumes **low-level constraints dominate**, but real-world texts may have strong **semantic or stylistic constraints**.

3. **Limited Constraint Types Used**  
   - Currently supports **mean, variance, skewness, kurtosis**.
   - Other constraint types (e.g., entropy-based metrics) could provide deeper insights.

---
"""


# ---- Step 1: Load Results ----
results_dir = "Results"
constraint_types = ["mean", "variance", "skewness", "kurtosis"]
report_path = os.path.join(results_dir, "Interpretation_Report.txt")

# Ensure results directory exists
if not os.path.exists(results_dir):
    raise FileNotFoundError(f"❌ Results directory '{results_dir}' not found. Run `train_best_maxent_train_test_split.py` first.")

# Find latest trained MaxEnt results
latest_constraint_type = None
for c_type in constraint_types:
    trained_dir = os.path.join(results_dir, f"Trained_MaxEnt_{c_type}")
    if os.path.exists(trained_dir):
        latest_constraint_type = c_type
        break

if not latest_constraint_type:
    raise FileNotFoundError(f"❌ No trained MaxEnt results found in `{results_dir}`.")

print(f"✅ Using latest trained MaxEnt results: {latest_constraint_type}")

# Define paths
trained_dir = os.path.join(results_dir, f"Trained_MaxEnt_{latest_constraint_type}")
lambda_file = os.path.join(trained_dir, "Trained_Lagrange_Multipliers.csv")
test_loss_file = os.path.join(trained_dir, "Test_Loss.txt")  # Assuming test loss is saved

# Load lambda values
if not os.path.exists(lambda_file):
    raise FileNotFoundError(f"❌ Lambda multipliers file `{lambda_file}` not found.")

lambda_df = pd.read_csv(lambda_file)

# Load test loss if available
test_loss = None
if os.path.exists(test_loss_file):
    with open(test_loss_file, "r") as f:
        test_loss = float(f.read().strip())

# ---- Step 2: Hierarchy-Based Interpretation ----
hierarchy = {
    "Character_Level": ["Zipf_Law", "Kolmogorov_Complexity"],
    "Morphology_Level": ["Syllable_Counts", "Word_Lengths", "Lexical_Diversity_TTR", "Lexical_Diversity_Shannon"],
    "Syntax_Level": ["POS_Tag_Distribution", "Avg_Sentence_Length", "Dependency_Complexity", "Punctuation_Usage"],
    "Grammar_Level": ["Readability_Dale_Chall", "Readability_Flesch_Kincaid", "Referential_Integrity"],
    "Semantic_Level": ["Cosine_Similarity", "WordNet_Similarity", "Lexical_Chains", "LDA_Topic_Consistency"],
    "Stylistic_Level": ["Style_Consistency", "BLEU_Score", "Jaccard_Score", "ROUGE_Score", "Text_Flow_Analysis"]
}

lambda_df["Level"] = lambda_df["Index"].map(
    {idx: level for level, indices in hierarchy.items() for idx in indices}
)

# Compute mean lambda per level
level_means = lambda_df.groupby("Level")["Lambda"].mean()
sorted_levels = list(hierarchy.keys())

# ---- Step 3: Generate Interpretation Report ----
with open(report_path, "w") as report:
    report.write(f"📊 **Maximum Entropy Model Interpretation Report**\n")
    report.write(f"----------------------------------------------\n\n")
    report.write(f"✅ Constraint Type Used: **{latest_constraint_type}**\n\n")
    
    if test_loss is not None:
        report.write(f"📉 **Test Loss**: {test_loss:.6f}\n")
        if test_loss < 0.05:
            report.write(f"✅ Test loss is **low** → The model generalizes well to unseen data.\n")
        elif test_loss < 0.2:
            report.write(f"⚠️ Test loss is **moderate** → Some generalization issues, might need more regularization.\n")
        else:
            report.write(f"❌ Test loss is **high** → Possible overfitting, check training data and constraints.\n")

    report.write("\n📌 **Constraint Strength per Linguistic Level**\n")
    for level in sorted_levels:
        strength = level_means.get(level, 0)
        report.write(f"- {level}: **{strength:.4f}**\n")
    
    # Hypothesis Check
    if level_means["Character_Level"] > level_means["Semantic_Level"]:
        report.write("\n✅ **Hypothesis Supported:** Lower levels impose stronger constraints.\n")
    else:
        report.write("\n❌ **Hypothesis Not Supported:** Higher-level constraints are unexpectedly stronger.\n")

    report.write("\n📈 **Detailed Lambda Values**\n")
    report.write(lambda_df.to_string(index=False))

print(f"✅ Interpretation report saved to `{report_path}`")

# ---- Step 4: Generate Plots ----
plot_dir = os.path.join(trained_dir, "Interpretation_Plots")
os.makedirs(plot_dir, exist_ok=True)

# 📌 Plot 1: Lagrange Multipliers Distribution
plt.figure(figsize=(12, 6))
sns.boxplot(x=lambda_df["Level"], y=lambda_df["Lambda"])
plt.xticks(rotation=45, ha="right")
plt.title(f"Lagrange Multipliers per Linguistic Level ({latest_constraint_type})")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Lagrange_Multipliers_Boxplot.png"), dpi=300)
plt.savefig(os.path.join(plot_dir, "Lagrange_Multipliers_Boxplot.pdf"), dpi=300)
plt.show()

# 📌 Plot 2: Constraint Strength per Level
plt.figure(figsize=(12, 6))
plt.bar(sorted_levels, [level_means[l] for l in sorted_levels], color='royalblue')
plt.xticks(rotation=30, ha="right", fontsize=12)
plt.title("Average Constraint Strength (Lagrange Multipliers) per Hierarchical Level")
plt.ylabel("Average Lambda Value")
plt.xlabel("Linguistic Hierarchy Level")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Average_Constraint_Strength.png"), dpi=300)
plt.savefig(os.path.join(plot_dir, "Average_Constraint_Strength.pdf"), dpi=300)
plt.show()

print(f"✅ Plots saved in `{plot_dir}/`")
