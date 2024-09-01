import json
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset

# Load the dataset
with open('merged_dataset.json', 'r') as f:
    dataset = json.load(f)

# Split the dataset into training and evaluation sets
def split_dataset(dataset, train_ratio=0.8):
    train_size = int(len(dataset) * train_ratio)
    train_data = dataset[:train_size]
    eval_data = dataset[train_size:]
    return train_data, eval_data

train_data, eval_data = split_dataset(dataset)

# Convert lists to Dataset objects
train_dataset = Dataset.from_dict({key: [d[key] for d in train_data] for key in train_data[0].keys()})
eval_dataset = Dataset.from_dict({key: [d[key] for d in eval_data] for key in eval_data[0].keys()})

# Load the pre-trained model and tokenizer
model_id = 'google/byt5-small'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,  # Use mixed precision
)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Concatenate relevant fields into a single text field
def concatenate_fields(examples):
    fields = [
        "Base notes", "Middle notes", "Top notes", "Notes", "Name", "Perfumer",
        "Description", "Gender", "Concepts", "URL", "All reviews", "Positive reviews", "Negative reviews"
    ]
    examples['text'] = ' '.join([examples[field] for field in fields if field in examples and examples[field]])
    return examples

train_dataset = train_dataset.map(concatenate_fields, batched=False)
eval_dataset = eval_dataset.map(concatenate_fields, batched=False)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Data collator for efficient data loading
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments with reduced batch size
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Adjust batch size as needed
    per_device_eval_batch_size=8,   # Adjust batch size as needed
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # Enable mixed precision training
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()