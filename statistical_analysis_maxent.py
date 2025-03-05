import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, kruskal, t
from statsmodels.stats.multitest import multipletests


"""
# 📊 **Statistical Analysis for Maximum Entropy Linguistic Constraints**
### **Author:** Enrico De Santis  
### **Version:** 1.0  
### **Last Updated:** [Date]  
### **Dependencies:** NumPy, Pandas, Seaborn, Matplotlib, SciPy, Statsmodels  

## **🔍 Overview**
This module performs **statistical analysis** on the Lagrange multipliers computed from the **Maximum Entropy Model** applied to linguistic constraints. It evaluates how different levels of linguistic hierarchy (character, morphology, syntax, grammar, semantics, and stylistics) influence text formation.

## **🎯 Objectives**
1. **Test Statistical Significance**  
   - Check whether linguistic levels impose significantly different constraints on text.
   - Use **ANOVA**, **Kruskal-Wallis**, and **t-tests** to compare levels.
   
2. **Measure Effect Sizes**  
   - Compute **Cohen’s d** to quantify the magnitude of differences.
   - Apply **Bonferroni correction** for multiple comparisons.

3. **Visualize Results**  
   - Generate **boxplots, barplots, and heatmaps** to understand constraint distributions.

4. **Generate a Summary Report**  
   - Save all statistical results in a `.txt` report.

---

## **📌 Methodology**
1. **Load Latest Trained Maximum Entropy Model Results**  
   - Finds the most recent trained model (`mean`, `variance`, `skewness`, or `kurtosis` constraints).  
   - Loads the estimated **Lagrange multipliers** from `Trained_Lagrange_Multipliers.csv`.  

2. **Define Linguistic Hierarchy**  
   - Groups constraints into hierarchical levels:
     - **Character-Level:** Zipf’s Law, Kolmogorov Complexity  
     - **Morphology-Level:** Syllables, Word Length, Lexical Diversity  
     - **Syntax-Level:** POS Tags, Sentence Length, Punctuation  
     - **Grammar-Level:** Readability Scores, Referential Integrity  
     - **Semantic-Level:** Cosine Similarity, WordNet Similarity  
     - **Stylistic-Level:** Style Consistency, BLEU, ROUGE, Jaccard  

3. **Perform Statistical Tests**  
   - **One-Way ANOVA:** Determines if means across levels differ significantly.  
   - **Kruskal-Wallis Test:** Non-parametric alternative to ANOVA for robustness.  
   - **Pairwise t-tests:** Compares every level pair to check significant differences.  
   - **Bonferroni Correction:** Adjusts p-values to prevent false positives.  
   - **Effect Size (Cohen’s d):** Measures practical significance of differences.  
   - **Confidence Intervals (95% CI):** Adds uncertainty estimation to effect sizes.  

4. **Generate Plots & Save Results**  
   - **Boxplot of Lagrange Multipliers:** Shows variance in constraint strength.  
   - **Barplot of Mean Constraint Strength:** Identifies the strongest constraints.  
   - **Heatmap of Cohen’s d:** Visualizes pairwise effect sizes.  
   - **Barplot of Cohen’s d with Confidence Intervals:** Highlights strongest differences.  

---

## **📌 Main Parameters & Their Meaning**
| Parameter | Description |
|-----------|------------|
| `results_dir` | Directory where results are stored (`Results/`). |
| `constraint_types` | List of constraints applied (`mean`, `variance`, etc.). |
| `trained_dir` | Path to the latest trained model directory. |
| `lambda_file` | CSV file containing estimated Lagrange multipliers. |
| `hierarchy` | Dictionary mapping linguistic constraints to hierarchical levels. |

---

## **📌 Outputs & File Structure**
| Output File | Description |
|------------|-------------|
| `Results/Statistical_Analysis_Report.txt` | Contains ANOVA, Kruskal-Wallis, t-tests, and effect sizes. |
| `Results/Trained_MaxEnt_[constraint]/Statistical_Plots/` | Folder for all plots. |
| `Effect_Size_Heatmap.png` | Heatmap of Cohen’s d effect sizes between levels. |
| `Lagrange_Multipliers_Boxplot.png` | Boxplot of constraint strength per level. |
| `Average_Constraint_Strength.png` | Barplot of mean constraint strength per level. |
| `Effect_Size_CI_Barplot.png` | Barplot of Cohen’s d with confidence intervals. |

---

## **📌 How to Interpret the Results**
1. **ANOVA & Kruskal-Wallis Results**  
   - If **p-value < 0.05**, at least one linguistic level is significantly different from others.  

2. **Pairwise t-tests & Effect Sizes**  
   - **p-value < 0.05 (Bonferroni Corrected):** Two levels differ significantly.  
   - **Cohen’s d > 0.8:** Strong difference.  
   - **Cohen’s d between 0.5 - 0.8:** Moderate difference.  
   - **Cohen’s d < 0.5:** Weak difference.  

3. **Visualizations**  
   - **Heatmap:** Darker colors indicate stronger effects.  
   - **Boxplot:** Levels with wider distributions have more variation.  
   - **Barplot:** Higher bars indicate stronger constraints at that level.  

---
"""


# ---- Step 1: Load Results ----
results_dir = "Results"
constraint_types = ["mean", "variance", "skewness", "kurtosis"]
report_path = os.path.join(results_dir, "Statistical_Analysis_Report.txt")

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

# Load lambda values
if not os.path.exists(lambda_file):
    raise FileNotFoundError(f"❌ Lambda multipliers file `{lambda_file}` not found.")

lambda_df = pd.read_csv(lambda_file)

# ---- Step 2: Define Linguistic Hierarchy ----
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

# ---- Step 3: Perform One-Way ANOVA ----
level_groups = [lambda_df[lambda_df["Level"] == level]["Lambda"].values for level in hierarchy.keys()]
anova_stat, anova_p = f_oneway(*level_groups)

# ---- Step 4: Perform Kruskal-Wallis Test ----
kruskal_stat, kruskal_p = kruskal(*level_groups)

# ---- Step 5: Perform Pairwise t-tests & Compute Effect Size (Cohen's d) ----
def compute_cohens_d(group1, group2):
    """Computes Cohen's d effect size between two groups."""
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)
    return mean_diff / pooled_std if pooled_std != 0 else 0

def compute_confidence_interval(group1, group2, confidence=0.95):
    """Computes confidence interval for Cohen's d using bootstrap resampling."""
    cohens_d = compute_cohens_d(group1, group2)
    n1, n2 = len(group1), len(group2)
    
    se = np.sqrt((1/n1) + (1/n2))  # Standard error of the difference
    df = min(n1, n2) - 1  # Degrees of freedom
    
    # Get t-critical value
    t_critical = t.ppf((1 + confidence) / 2, df)
    
    # Compute confidence interval
    margin_of_error = t_critical * se
    ci_lower, ci_upper = cohens_d - margin_of_error, cohens_d + margin_of_error
    
    return cohens_d, ci_lower, ci_upper

pairwise_results = []
levels = list(hierarchy.keys())

for i in range(len(levels)):
    for j in range(i + 1, len(levels)):
        group1 = lambda_df[lambda_df["Level"] == levels[i]]["Lambda"].values
        group2 = lambda_df[lambda_df["Level"] == levels[j]]["Lambda"].values

        stat, p_value = ttest_ind(group1, group2)
        cohens_d, ci_lower, ci_upper = compute_confidence_interval(group1, group2)

        pairwise_results.append((levels[i], levels[j], stat, p_value, cohens_d, ci_lower, ci_upper))

# Apply Bonferroni Correction
p_values = [res[3] for res in pairwise_results]
reject, p_adjusted, _, _ = multipletests(p_values, method="bonferroni")

# Store results
pairwise_df = pd.DataFrame(pairwise_results, columns=["Level_1", "Level_2", "t_statistic", "p_value", "Cohen_d", "CI_Lower", "CI_Upper"])
pairwise_df["p_adjusted"] = p_adjusted
pairwise_df["Significant"] = reject

# ---- Step 6: Generate Interpretation Report ----
with open(report_path, "w") as report:
    report.write(f"📊 **Statistical Analysis Report for Maximum Entropy Model**\n")
    report.write(f"---------------------------------------------------------\n\n")
    report.write(f"✅ Constraint Type Used: **{latest_constraint_type}**\n\n")

    report.write(f"📈 **ANOVA Test Result:**\n")
    report.write(f"  - F-statistic: {anova_stat:.4f}\n")
    report.write(f"  - p-value: {anova_p:.6f}\n")

    report.write(f"📈 **Kruskal-Wallis Test Result:**\n")
    report.write(f"  - H-statistic: {kruskal_stat:.4f}\n")
    report.write(f"  - p-value: {kruskal_p:.6f}\n")

    report.write("\n📉 **Pairwise t-tests with Bonferroni Correction & Cohen's d Effect Size (95% CI):**\n")
    report.write(pairwise_df.to_string(index=False))

print(f"✅ Statistical report saved to `{report_path}`")

# ---- Step 7: Generate Plots ----
plot_dir = os.path.join(trained_dir, "Statistical_Plots")
os.makedirs(plot_dir, exist_ok=True)

# 📌 Heatmap: Effect Size (Cohen's d) for Pairwise Level Comparisons
effect_size_matrix = pairwise_df.pivot(index="Level_1", columns="Level_2", values="Cohen_d")

plt.figure(figsize=(10, 8))
sns.heatmap(effect_size_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Effect Size (Cohen's d) for Pairwise Level Comparisons")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Effect_Size_Heatmap.png"), dpi=300)
plt.savefig(os.path.join(plot_dir, "Effect_Size_Heatmap.pdf"), dpi=300)
plt.show()


# 📌 Boxplot of Lagrange Multipliers per Level
plt.figure(figsize=(12, 6))
sns.boxplot(x=lambda_df["Level"], y=lambda_df["Lambda"])
plt.xticks(rotation=45, ha="right")
plt.title(f"Lagrange Multipliers per Linguistic Level ({latest_constraint_type})")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Lagrange_Multipliers_Boxplot.png"), dpi=300)
plt.savefig(os.path.join(plot_dir, "Lagrange_Multipliers_Boxplot.pdf"), dpi=300)
plt.show()

# 📌 Barplot of Mean Constraint Strength per Level
level_means = lambda_df.groupby("Level")["Lambda"].mean()

plt.figure(figsize=(12, 6))
plt.bar(levels, [level_means[l] for l in levels], color='royalblue')
plt.xticks(rotation=30, ha="right", fontsize=12)
plt.title("Average Constraint Strength (Lagrange Multipliers) per Hierarchical Level")
plt.ylabel("Average Lambda Value")
plt.xlabel("Linguistic Hierarchy Level")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Average_Constraint_Strength.png"), dpi=300)
plt.savefig(os.path.join(plot_dir, "Average_Constraint_Strength.pdf"), dpi=300)
plt.show()

# 📌 Barplot: Cohen's d with Confidence Intervals
plt.figure(figsize=(12, 6))
sns.barplot(x="Level_1", y="Cohen_d", data=pairwise_df, capsize=0.1, ci=None)
plt.errorbar(x=pairwise_df["Level_1"], y=pairwise_df["Cohen_d"], yerr=[pairwise_df["Cohen_d"] - pairwise_df["CI_Lower"], pairwise_df["CI_Upper"] - pairwise_df["Cohen_d"]], fmt='o', color='black', capsize=5)
plt.xticks(rotation=45, ha="right")
plt.title("Effect Size (Cohen's d) with 95% Confidence Intervals")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Effect_Size_CI_Barplot.png"), dpi=300)
plt.savefig(os.path.join(plot_dir, "Effect_Size_CI_Barplot.pdf"), dpi=300)
plt.show()


print(f"✅ Statistical plots saved in `{plot_dir}/`")
