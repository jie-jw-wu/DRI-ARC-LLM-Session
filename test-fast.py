import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy("mixed_float16")  # Use mixed precision

# Use a smaller dataset and shorter sequences

# Load and Encode Dataset
data = pd.read_csv("poynter_data.csv")  # Replace with your file path
data = data.sample(n=500, random_state=42)
encoder = LabelEncoder()
data['label'] = encoder.fit_transform(data['label'])

# Initialize BERT Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
#model.bert.gradient_checkpointing_enable()

# Tokenize Texts
texts = data['text'].tolist()
inputs = tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors="np")  # Use "np" for NumPy
labels = data['label'].values  # Convert labels to a NumPy array directly

# Split Data into Training and Validation Sets with train_test_split
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    inputs['input_ids'], labels, test_size=0.1, random_state=42
)

# Convert to TensorFlow Dataset Objects
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels)).batch(8)
val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels)).batch(8)

# Compile the Model
num_train_steps = len(train_dataset) * 1  # Assuming 3 epochs
optimizer, _ = create_optimizer(init_lr=5e-5, num_train_steps=num_train_steps, num_warmup_steps=0)

#for layer in model.bert.layers:
#    layer.trainable = False

model.compile(optimizer=optimizer, 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=["accuracy"])

# Train the Model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=1  # Adjust based on validation performance
)

# Save the Fine-Tuned Model
model.save_pretrained("covid_bert_model")