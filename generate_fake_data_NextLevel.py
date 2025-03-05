import numpy as np
import pandas as pd
import os
import random

"""
# 📊 **Fake Linguistic Dataset Generator for Maximum Entropy Modeling**
### **Author:** Enrico De Santis 
### **Version:** 1.0  
### **Last Updated:** [Date]  
### **Dependencies:** NumPy, Pandas, OS, Random  

---

## **🔍 Overview**
This script generates a **synthetic dataset** designed for experiments in **linguistic modeling**  
and **maximum entropy analysis**. It **creates random textual data** along with **simulated linguistic indices**,  
normalizes the dataset, and stores it for later processing.

---

## **🎯 Objectives**
1. **Generate Fake Text Data**  
   - Randomized **word sequences** to simulate **linguistic corpora**.
   - Assigns **random classification labels** (e.g., **categories** or **genres**).

2. **Generate Fake Linguistic Features**  
   - Includes **various text complexity measures**, such as:
     - **Lexical, Syntactic, Semantic, and Stylistic indices.**
   - **Values are randomly generated** to simulate numerical text features.

3. **Normalize the Dataset**  
   - Standardizes features using **z-score normalization**.

4. **Save the Dataset in CSV Format**  
   - Stores **original and normalized** datasets in `Data/`.

---

## **📌 Methodology**
### **Step 1: Create Output Directory**
- Ensures that the `Data/` directory exists.

### **Step 2: Define Dataset Properties**
- **Number of samples** = `121`  
- **Number of classification labels** = `3`  
- **Feature set** includes:
  - **Zipf's Law metrics**
  - **Kolmogorov Complexity**
  - **Lexical Diversity Metrics**
  - **Syntactic & Grammatical Complexity**
  - **Semantic & Stylistic Measures**

### **Step 3: Generate Fake Text Data**
- Uses a **word pool** to create **synthetic sentences**.
- Text length ranges between **10 and 20 words**.
- **Random labels (1, 2, or 3)** are assigned to each text.

### **Step 4: Generate Numerical Index Data**
- **Random values** between `[0,1]` simulate linguistic features.

### **Step 5: Normalize the Dataset**
- **Z-score normalization** is applied to **all numerical features**.

### **Step 6: Save the Data**
- Two CSV files are generated:
  1. **Raw dataset:** `fake_data_nextLevel.csv`
  2. **Normalized dataset:** `fake_data_nextLevel_normalized.csv`

---

## **📌 Main Parameters & Their Meaning**
| Parameter | Description |
|-----------|------------|
| `num_texts` | Number of generated text samples (default `121`). |
| `num_classes` | Number of classification labels (default `3`). |
| `index_columns` | List of generated linguistic indices. |
| `fake_labels` | Randomly assigned labels (`1`, `2`, or `3`). |
| `fake_texts` | Generated sentences using **random words**. |

---

## **📌 Outputs & File Structure**
| Output File | Description |
|------------|-------------|
| `Data/fake_data_nextLevel.csv` | Original dataset (raw feature values). |
| `Data/fake_data_nextLevel_normalized.csv` | Normalized dataset (z-score applied). |

---

## **📌 How to Interpret the Dataset**
| Feature Type | Example Features | Interpretation |
|-------------|----------------|---------------|
| **Lexical** | Zipf_Law, Word_Lengths | Word frequency & complexity |
| **Syntactic** | POS_Tag_Distribution, Avg_Sentence_Length | Sentence structure analysis |
| **Grammatical** | Readability_Flesch_Kincaid, Referential_Integrity | Grammar complexity |
| **Semantic** | Cosine_Similarity, WordNet_Similarity | Meaning relationships |
| **Stylistic** | BLEU_Score, ROUGE_Score, Text_Flow_Analysis | Writing style consistency |

---

## **📌 Limitations**
1. **Randomness of Generated Texts**  
   - The text samples **do not contain real linguistic structure**.  
   - Future work: **Use a language model (e.g., GPT) to generate coherent text**.

2. **Feature Simulation**  
   - The features **do not reflect true linguistic complexity**.  
   - Future work: **Extract real linguistic features from natural corpora**.

3. **Fixed Number of Classes**  
   - The dataset currently supports **only 3 labels**.  
   - Future work: **Increase label diversity and add metadata annotations**.

---

"""

# ---- Step 1: Create Output Directory ----
data_dir = "Data"
os.makedirs(data_dir, exist_ok=True)

# ---- Step 2: Define Dataset Properties ----
np.random.seed(42)
random.seed(42)

num_texts = 121
num_classes = 3  # Example: 3 classes for classification

index_columns = [
    "Zipf_Law", "Kolmogorov_Complexity",
    "Syllable_Counts", "Word_Lengths", "Lexical_Diversity_TTR", "Lexical_Diversity_Shannon",
    "POS_Tag_Distribution", "Avg_Sentence_Length", "Dependency_Complexity", "Punctuation_Usage",
    "Readability_Dale_Chall", "Readability_Flesch_Kincaid", "Referential_Integrity",
    "Cosine_Similarity", "WordNet_Similarity", "Lexical_Chains", "LDA_Topic_Consistency",
    "Style_Consistency", "BLEU_Score", "Jaccard_Score", "ROUGE_Score", "Text_Flow_Analysis"
]

# ---- Step 3: Generate Fake Text Data ----
# Define a set of words to create simple fake texts
words_pool = ["AI", "language", "model", "entropy", "learning", "text", "information",
              "analysis", "structure", "pattern", "semantic", "syntax", "data", "complexity"]

def generate_fake_text():
    """Generate a simple fake text with random words."""
    return " ".join(random.choices(words_pool, k=random.randint(10, 20)))  # 10 to 20 words per text

# Create a list of fake texts
fake_texts = [generate_fake_text() for _ in range(num_texts)]

# Create class labels (randomly assigned)
fake_labels = [random.choice(range(1, num_classes + 1)) for _ in range(num_texts)]

# ---- Step 4: Generate Numerical Index Data ----
fake_data = np.random.rand(num_texts, len(index_columns))  # Fake values between 0 and 1

# ---- Step 5: Create DataFrame ----
df = pd.DataFrame(fake_data, columns=index_columns)
df.insert(0, "Label", fake_labels)  # Add class label as first column
df.insert(1, "Text", fake_texts)  # Add text as second column

# ---- Step 6: Normalize the Index Columns ----
df_normalized = df.copy()
df_normalized[index_columns] = (df[index_columns] - df[index_columns].mean()) / df[index_columns].std()

# ---- Step 7: Save the Data ----
original_data_path = os.path.join(data_dir, "fake_data_nextLevel.csv")
normalized_data_path = os.path.join(data_dir, "fake_data_nextLevel_normalized.csv")

df.to_csv(original_data_path, index=False)
df_normalized.to_csv(normalized_data_path, index=False)

print(f"✅ Fake dataset with texts and labels created and saved in '{data_dir}/'.")
print(f"   - Original data: {original_data_path}")
print(f"   - Normalized data: {normalized_data_path}")
