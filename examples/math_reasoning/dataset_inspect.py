import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# 1. Load the dataset from Hugging Face
# We only need the 'train' split as this dataset is typically used for training
print("Loading dataset...")
dataset = load_dataset("zwhe99/DeepMath-103K", split="train")

# 2. Convert to Pandas DataFrame for easier analysis
# The dataset has columns: ['question', 'final_answer', 'difficulty', 'topic', ...]
df = pd.DataFrame(dataset)

# 3. Compute Difficulty Distribution
# We round to the nearest whole number or 0.5 step if preferred, 
# but here we analyze the raw values first.
difficulty_counts = df['difficulty'].value_counts().sort_index()

# 4. Print Summary Statistics
print("\n--- Difficulty Statistics ---")
print(df['difficulty'].describe())

print("\n--- Distribution (Top 10 most common levels) ---")
print(difficulty_counts)

# 5. (Optional) Visualize the Distribution
plt.figure(figsize=(10, 6))
# Create bins for integer levels (1, 2, ..., 10)
plt.hist(df['difficulty'], bins=range(1, 12), edgecolor='black', alpha=0.7, align='left')
plt.title("Difficulty Distribution of DeepMath-103K")
plt.xlabel("Difficulty Level")
plt.ylabel("Number of Problems")
plt.xticks(range(1, 11))
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Save the plot or show it
# plt.savefig("deepmath_distribution.png")
plt.show()