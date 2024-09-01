from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import torch
import os

# Define a custom data collator to adjust the weights of each attribute
class WeightedDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, mlm_probability=0.15, weights=None):
        super().__init__(tokenizer, mlm, mlm_probability)
        self.weights = weights if weights else {}

    def __call__(self, examples):
        batch = super().__call__(examples)
        if self.weights:
            for key, weight in self.weights.items():
                if key in batch:
                    batch[key] = batch[key] * weight
        return batch

def main():
    # Load the dataset
    dataset = load_dataset('json', data_files='merged_dataset.json', split='train')

    # Split the dataset into training and evaluation sets
    def split_dataset(dataset, test_size=0.01):
        train_data = []
        eval_data = []
        for i, example in enumerate(dataset):
            if i % int(1 / test_size) == 0:
                eval_data.append(example)
            else:
                train_data.append(example)
        return train_data, eval_data

    train_data, eval_data = split_dataset(dataset)

    # Convert lists to Dataset objects
    train_dataset = Dataset.from_dict({key: [d[key] for d in train_data] for key in train_data[0].keys()})
    eval_dataset = Dataset.from_dict({key: [d[key] for d in eval_data] for key in eval_data[0].keys()})

    # Load the pre-trained model and tokenizer
    model_id = "google/byt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,  # Use mixed precision
    )

    # Set pad_token to eos_token if pad_token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient checkpointing to save memory
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

    # Clear CUDA cache frequently
    torch.cuda.empty_cache()

    # Define weights for each attribute
    attribute_weights = {
        "Base notes": 10.0,
        "Middle notes": 7.0,
        "Top notes": 7.0,
        "Notes": 20.0,
        "Name": 1.3,
        "Perfumer": 1.0,
        "Description": 10.0,
        "Gender": 1.0,
        "Concepts": 20.0,
        "URL": 1.0,
        "All reviews": 8.5,
        "Positive reviews": 5.0,
        "Negative reviews": 5.0
    }

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,  # Reduce batch size
        per_device_eval_batch_size=1,   # Reduce batch size
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        gradient_checkpointing=True,     # Enable gradient checkpointing
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        fp16=False,                      # Disable mixed precision training
        dataloader_pin_memory=True,     # Pin memory for DataLoader
        dataloader_num_workers=2,       # Number of workers for DataLoader
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=WeightedDataCollator(tokenizer, weights=attribute_weights),
    )

    # Train the model
    trainer.train()

    # Save the model, tokenizer, and training arguments
    output_dir = "./final_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    training_args.save(output_dir)

if __name__ == '__main__':
    main()