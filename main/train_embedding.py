import json
import torch
from transformers import AutoTokenizer
from datasets import Dataset
from sentence_transformers import SentenceTransformer

# Load the dataset
with open('merged_dataset.json', 'r') as f:
    dataset = json.load(f)

# Split the dataset into training and evaluation sets
def split_dataset(dataset, train_ratio=0.98):
    train_size = int(len(dataset) * train_ratio)
    train_data = dataset[:train_size]
    eval_data = dataset[train_size:]
    return train_data, eval_data

train_data, eval_data = split_dataset(dataset)

# Convert lists to Dataset objects
train_dataset = Dataset.from_dict({key: [d[key] for d in train_data] for key in train_data[0].keys()})
eval_dataset = Dataset.from_dict({key: [d[key] for d in eval_data] for key in eval_data[0].keys()})

# Load the pre-trained tokenizer and model
model_id = 'mixedbread-ai/mxbai-embed-large-v1'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = SentenceTransformer(model_id).cuda()

# Define weights for each attribute
attribute_weights = {
    "Base notes": 5.0,
    "Middle notes": 3.5,
    "Top notes": 3.5,
    "Notes": 10.0,
    "Name": 1.0,
    "Perfumer": 1.5,
    "Description": 7.5,
    "Gender": 1.0, 
    "Concepts": 15.0, 
    "URL": 1.0,
    "All reviews": 10.0,
    "Positive reviews": 7.0,
    "Negative reviews": 7.0
}

# Concatenate relevant fields into a single text field with weights
def concatenate_fields(examples):
    fields = [
        "Base notes", "Middle notes", "Top notes", "Notes", "Name", "Perfumer",
        "Description", "Gender", "Concepts", "URL", "All reviews", "Positive reviews", "Negative reviews"
    ]
    examples['text'] = ' '.join([examples[field] * int(attribute_weights[field]) for field in fields if field in examples and examples[field]])
    return examples

train_dataset = train_dataset.map(concatenate_fields, batched=False)
eval_dataset = eval_dataset.map(concatenate_fields, batched=False)

# Extract embeddings using SentenceTransformer
def extract_embeddings(dataset):
    embeddings = []
    for example in dataset:
        text = example['text']
        embedding = model.encode(text, convert_to_tensor=True)
        embeddings.append(embedding)
    return torch.stack(embeddings)

train_embeddings = extract_embeddings(train_dataset)
eval_embeddings = extract_embeddings(eval_dataset)

# Save the embeddings using torch
torch.save(train_embeddings, "train_embeddings.pt")
torch.save(eval_embeddings, "eval_embeddings.pt")

# Example usage: Load the embeddings
train_embeddings = torch.load("train_embeddings.pt")
eval_embeddings = torch.load("eval_embeddings.pt")
model.save_pretrained(r'C:\Users\default.DESKTOP-7FKFEEG\project\model2')
tokenizer.save_pretrained(r'C:\Users\default.DESKTOP-7FKFEEG\project\tokenizer2')
print(train_embeddings.shape)
print(eval_embeddings.shape)