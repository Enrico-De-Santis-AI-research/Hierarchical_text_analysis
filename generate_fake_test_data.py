import numpy as np
import pandas as pd
import os

# ---- Step 1: Define Data Parameters ----
np.random.seed(999)  # Ensure different data from training set
num_test_texts = 50  # Smaller test set

index_columns = [
    "Zipf_Law", "Kolmogorov_Complexity",
    "Syllable_Counts", "Word_Lengths", "Lexical_Diversity_TTR", "Lexical_Diversity_Shannon",
    "POS_Tag_Distribution", "Avg_Sentence_Length", "Dependency_Complexity", "Punctuation_Usage",
    "Readability_Dale_Chall", "Readability_Flesch_Kincaid", "Referential_Integrity",
    "Cosine_Similarity", "WordNet_Similarity", "Lexical_Chains", "LDA_Topic_Consistency",
    "Style_Consistency", "BLEU_Score", "Jaccard_Score", "ROUGE_Score", "Text_Flow_Analysis"
]

# ---- Step 2: Generate Random Feature Data ----
fake_test_data = np.random.rand(num_test_texts, len(index_columns))  # Fake values between 0 and 1

# ---- Step 3: Generate Fake Labels and Texts ----
fake_labels = np.random.choice(["Class_A", "Class_B"], size=num_test_texts)  # Random classification labels
fake_texts = [f"Fake text sample {i+1}" for i in range(num_test_texts)]  # Placeholder texts

# ---- Step 4: Construct the DataFrame ----
df_test = pd.DataFrame(fake_test_data, columns=index_columns)
df_test.insert(0, "Text", fake_texts)
df_test.insert(0, "Label", fake_labels)

# ---- Step 5: Normalize Only the Numerical Columns ----
train_data_path = "Data/fake_data_nextLevel_normalized.csv"

if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"❌ Training data not found. Run 'generate_fake_data_nextLevel.py' first.")

df_train = pd.read_csv(train_data_path)

# Identify numerical columns for normalization
numerical_cols = [col for col in df_train.columns if col not in ["Label", "Text"]]

# Normalize using training data statistics
df_test[numerical_cols] = (df_test[numerical_cols] - df_train[numerical_cols].mean()) / df_train[numerical_cols].std()

# ---- Step 6: Save the Test Dataset ----
test_data_path = "Data/fake_data_test_normalized.csv"
df_test.to_csv(test_data_path, index=False)
print(f"✅ Fake test dataset saved to {test_data_path}")
