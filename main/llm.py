from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load the tokenizer and model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Path to the JSON file
json_file_path = r'C:\Users\default.DESKTOP-7FKFEEG\project\frag\scraper\dataset.json'

# Load the dataset
dataset = load_dataset('json', data_files=json_file_path)

# Print the column names to debug
print(dataset['train'].column_names)

def preprocess_function(examples):
    # Combine the notes and URL into a single string for training
    inputs = [f"Name: {name}\nNotes: {note}\nBase notes: {base}\nMiddle notes: {middle}\nTop notes: {top}\nReviews: {reviews}\nPositive Reviews: {previews}\nNegative Reviews: {nreviews}\nURL: {url}" for name, note, base, middle, top, reviews, previews, nreviews, url in zip(examples['Name'], examples['Notes'], examples['Base notes'], examples['Middle notes'], examples['Top notes'], examples['All reviews'], examples['Positive reviews'], examples['Negative reviews'], examples['URL'])]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    # Add decoder_input_ids using text_target
    labels = tokenizer(text_target=inputs, padding="max_length", truncation=True, max_length=512)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the preprocessing function to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Remove columns that are not needed
tokenized_dataset = tokenized_dataset.remove_columns(["Base notes", "Middle notes", "Top notes", "Notes", "URL"])

# Split the dataset into train and test sets
train_test_split = tokenized_dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Set the format for PyTorch
train_dataset.set_format("torch")
test_dataset.set_format("torch")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained(r"C:\Users\default.DESKTOP-7FKFEEG\project\model")
tokenizer.save_pretrained(r"C:\Users\default.DESKTOP-7FKFEEG\project\model")