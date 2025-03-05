# 📊 Linguistic Feature Analysis using Maximum Entropy & Neural Network Dependencies

## **🚀 Overview**
This project focuses on **analyzing linguistic structures** using **Maximum Entropy models** with **Lagrange multipliers** and integrating **neural network-based feature dependencies** to regularize constraints.

### **Key Features:**
- **Synthetic dataset generation** for linguistic indices.
- **Feature hierarchy analysis** across lexical, syntactic, semantic, and stylistic dimensions.
- **Maximum Entropy training** to estimate Lagrange multipliers using different constraints (mean, variance, skewness, kurtosis).
- **Neural Network regularization** to enforce **nonlinear dependencies** between features.
- **Statistical validation** (ANOVA, t-tests, effect size) to evaluate constraint significance.
- **Visualization and analysis** using heatmaps, boxplots, and correlation matrices.

---

## **📂 Project Structure**
```
📦 Linguistic-Entropy-Model
│
├── 📂 Data                           # 🔹 Contains synthetic & processed datasets
│   ├── fake_data_nextLevel.csv
│   ├── fake_data_nextLevel_normalized.csv
│   ├── fake_data_test_normalized.csv
│
├── 📂 Models                         # 🔹 Trained Neural Network models for feature dependencies
│   ├── dependency_nn_best.pth
│
├── 📂 Results                        # 🔹 Stores results from Maximum Entropy training & grid search
│   ├── Grid_Search_mean/
│   ├── Trained_MaxEnt_mean/
│   ├── Statistical_Plots/
│   ├── NN_Feature_Dependencies.csv
│   ├── Lagrange_Multipliers.csv
│
├── 📂 Scripts                        # 🔹 Core scripts for training, testing, and visualization
│   ├── generate_fake_data_nextLevel.py
│   ├── generate_fake_test_data.py
│   ├── train_dependency_nn.py
│   ├── train_best_maxent_train_test_split.py
│   ├── maxent_grid_search.py
│   ├── statistical_analysis.py
│   ├── interpret_results.py
│
├── README.md                         # 🔹 Project documentation (this file)
└── requirements.txt                   # 🔹 Required Python dependencies
```

---

## **📌 Features & Modules**
### **🔹 1. Synthetic Dataset Generation**
| **Script** | `generate_fake_data_nextLevel.py` & `generate_fake_test_data.py` |
|------------|---------------------------------------------------------------|
| **Purpose** | Generates **synthetic linguistic datasets** with numerical linguistic indices. |
| **Features** | - Randomly assigns **classification labels**. <br> - Creates **linguistic index values** (Zipf Law, POS Distribution, Readability scores, etc.). <br> - Normalizes the dataset for **model training**. |
| **Output Files** | - `fake_data_nextLevel_normalized.csv` (Training set) <br> - `fake_data_test_normalized.csv` (Test set) |

### **🔹 2. Neural Network for Feature Dependencies**
| **Script** | `train_dependency_nn.py` |
|------------|--------------------------|
| **Purpose** | Trains a **Neural Network (NN)** to learn dependencies between linguistic features. |
| **Architecture** | - 3-layer fully connected NN. <br> - Uses **ReLU activations** and **Adam optimizer**. |
| **Feature Dependencies** | - Computes similarity-based penalties between features. <br> - Helps **regularize Maximum Entropy training**. |
| **Output Files** | `dependency_nn_best.pth` (Best trained model) |

### **🔹 3. Maximum Entropy Model Training**
| **Script** | `train_best_maxent_train_test_split.py` |
|------------|------------------------------------------|
| **Purpose** | Uses **Maximum Entropy principles** to estimate linguistic constraints. |
| **Key Concepts** | - **Lagrange multipliers** to enforce constraints. <br> - Supports **four constraints:** Mean, Variance, Skewness, Kurtosis. <br> - Optimized using **L-BFGS-B**. |
| **Neural Regularization** | - Uses the **pre-trained NN** to regularize constraints between features. |
| **Output Files** | `Lagrange_Multipliers.csv` |

### **🔹 4. Grid Search for Best Hyperparameters**
| **Script** | `maxent_grid_search.py` |
|------------|-------------------------|
| **Purpose** | Finds the **best regularization hyperparameters** for Maximum Entropy training. |
| **Hyperparameters** | - `Alpha` (L2 Regularization). <br> - `Beta` (Neural Network dependency penalty). |
| **Parallelization** | - Uses **multiprocessing** for fast grid search. |
| **Output Files** | `Grid_Search_Results.csv`, `Grid_Search_Loss_Heatmap.png` |

### **🔹 5. Statistical Significance Tests**
| **Script** | `statistical_analysis.py` |
|------------|----------------------------|
| **Purpose** | Tests **statistical significance** of constraints across linguistic levels. |
| **Tests Performed** | - **One-Way ANOVA** (Checks if constraint strengths differ across levels). <br> - **Kruskal-Wallis Test** (Non-parametric alternative). <br> - **Pairwise t-tests with Bonferroni correction**. <br> - **Effect size (Cohen’s d)**. |
| **Output Files** | `Statistical_Analysis_Report.txt` |

### **🔹 6. Interpretation of Results**
| **Script** | `interpret_results.py` |
|------------|------------------------|
| **Purpose** | **Summarizes findings** from Maximum Entropy training. |
| **Key Insights** | - Which linguistic levels have **stronger constraints**? <br> - Does **feature hierarchy** hold (Character-Level > Morphology > Syntax)? |
| **Output Files** | `Interpretation_Report.txt` |

---

## **📊 Visualizations**
- **Boxplot of Lagrange Multipliers** (`Lagrange_Multipliers_Boxplot.png`)
- **Heatmap of Feature Dependencies** (`NN_Feature_Dependencies_Heatmap.png`)
- **Effect Size Barplot** (`Effect_Size_CI_Barplot.png`)
- **Grid Search Heatmap** (`Grid_Search_Loss_Heatmap.png`)

---

## **📌 Installation**
```bash
# Clone the repository
git clone https://github.com/your-repo-name.git
cd Linguistic-Entropy-Model

# Install dependencies
pip install -r requirements.txt
```

---

## **📌 Usage**
### **1️⃣ Generate Synthetic Data**
```bash
python generate_fake_data_nextLevel.py
python generate_fake_test_data.py
```
### **2️⃣ Train Neural Network for Feature Dependencies**
```bash
python train_dependency_nn.py
```
### **3️⃣ Train Maximum Entropy Model**
```bash
python train_best_maxent_train_test_split.py
```
### **4️⃣ Run Hyperparameter Grid Search**
```bash
python maxent_grid_search.py
```
### **5️⃣ Perform Statistical Analysis**
```bash
python statistical_analysis.py
```
### **6️⃣ Generate Interpretation Report**
```bash
python interpret_results.py
```

---

## **📜 License**
This project is licensed under the **MIT License**.

---

## **👨‍💻 Author**
[Your Name]  
🔗 [Your Website or GitHub Profile]  
📩 *For collaborations, feel free to reach out!* 🚀

