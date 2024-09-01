import json

# Load dataset.json with UTF-8 encoding
with open('merged_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Load wiki_parfum.json with UTF-8 encoding
with open('nst.json', 'r', encoding='utf-8') as f:
    wiki_parfum = json.load(f)

# Create a dictionary for quick lookup by name
wiki_parfum_dict = {entry['Name']: entry for entry in wiki_parfum}

# Define all possible attributes
all_attributes = [
    "Base notes", "Middle notes", "Top notes", "Notes", "Name", "Perfumer",
    "Description", "Gender", "Concepts", "URL", "All reviews", "Positive reviews", "Negative reviews", "review"
]

# Merge datasets
merged_data = []

for entry in dataset:
    name = entry.get('Name')
    if name in wiki_parfum_dict:
        # Merge the entries
        merged_entry = {attr: entry.get(attr, "") for attr in all_attributes}
        wiki_entry = wiki_parfum_dict[name]
        for attr in ["Perfumer", "Description", "Gender", "Concepts"]:
            merged_entry[attr] = wiki_entry.get(attr, "")
        merged_data.append(merged_entry)
    else:
        # Add the entry as is, ensuring all attributes are present
        merged_entry = {attr: entry.get(attr, "") for attr in all_attributes}
        merged_data.append(merged_entry)

# Add remaining entries from wiki_parfum that were not in dataset
for name, entry in wiki_parfum_dict.items():
    if name not in [e['Name'] for e in dataset]:
        merged_entry = {attr: entry.get(attr, "") for attr in all_attributes}
        merged_data.append(merged_entry)

# Save the merged data to a new JSON file
with open('merge.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=4)

print("Merging complete. The merged data is saved in 'merged_dataset.json'.")