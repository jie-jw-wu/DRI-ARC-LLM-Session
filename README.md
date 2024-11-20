# Fine-Tuning a Language Model with TensorFlow and Transformers

This repository provides a demonstration of fine-tuning a pre-trained language model (BERT) for a text classification task using TensorFlow, Hugging Face Transformers, and Scikit-learn. The provided scripts and job scheduler instructions enable efficient fine-tuning on a CPU-based environment, such as an HPC cluster.

Slides: https://docs.google.com/presentation/d/1eNdsWbLccc_8GnwWgOo2u6U5Z05eswu0S30Z03TqnPU/edit?usp=sharing

---

## Prerequisites

### Modules and Environment
Ensure you have access to the following modules or packages:
- Python 3.10 or higher
- TensorFlow
- Transformers
- Scikit-learn
- Pandas

### Dataset
- A CSV file (`poynter_data.csv`) with two columns:
  - `text`: The input text data for classification.
  - `label`: Corresponding labels for the text.

---

## Workflow

### Python Fine-Tuning Script

The fine-tuning process involves:
1. **Loading the Dataset**: The dataset is read from a CSV file and labels are encoded using `LabelEncoder`.
2. **Tokenizing Text Data**: The BERT tokenizer processes the input text to prepare it for the model.
3. **Splitting Data**: The data is split into training and validation sets using `train_test_split`.
4. **Dataset Conversion**: Input data and labels are converted into TensorFlow dataset objects.
5. **Model Compilation**: A BERT model is initialized with a sequence classification head and compiled with a custom optimizer.
6. **Model Training**: The model is trained for three epochs with a batch size of 16.
7. **Saving the Model**: The fine-tuned model is saved for future inference or deployment.

Run the script using (submit a batch job):
```bash
sbatch script_test_cpu.sh
