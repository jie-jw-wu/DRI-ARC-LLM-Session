# Hugging Face Dataset Visualization Notebook

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# Load dataset
# Here, we use the "emotion" dataset from Hugging Face.
dataset = load_dataset("emotion")

# Display a sample from the training set
print("\nSample data from the training set:")
print(dataset['train'].to_pandas().head())

# Visualize class distribution
def visualize_class_distribution(dataset, split):
    """
    Visualizes the class distribution for the given dataset and split.

    Args:
        dataset: The Hugging Face dataset object.
        split: The specific split to analyze (e.g., 'train').
    """
    df = dataset[split].to_pandas()
    class_distribution = df['label'].value_counts()
    print("\nClass Distribution:")
    print(class_distribution)

    # Map numerical labels to textual labels
    class_names = dataset['train'].features['label'].names
    class_distribution.index = [class_names[i] for i in class_distribution.index]

    # Plot distribution
    plt.figure(figsize=(8, 6))
    class_distribution.plot(kind="bar", color="skyblue")
    plt.title(f"Class Distribution for {split} Split")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

# Visualize for the training set
visualize_class_distribution(dataset, "train")
