from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.nn import BCEWithLogitsLoss

# Define the new labels
label_names = ["misandrist", "misogynist",] # misandrists hate men. misogynists hate women.
num_labels = len(label_names)

# Load the BERT tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Assume you have a dataset with two lists: texts (your tweets) and multi-labels
texts = ["tweet1", "tweet2", "tweet3", "tweet4"]
labels = [[1, 0], [0, 1], [0, 0]] # [1, 0] for misandrist, [0, 1] for misogynist, [0, 0] for neither.

# Tokenize the dataset
inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
inputs['labels'] = torch.tensor(labels)

# Split the dataset into training and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(inputs['input_ids'], inputs['labels'], test_size=0.2)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
)

# Define the loss function
loss_fn = BCEWithLogitsLoss()

# Define the training process
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_inputs,
    eval_dataset=val_inputs,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
    tokenizer=tokenizer,
    loss_fn=loss_fn,
)

# Train the model
trainer.train()
