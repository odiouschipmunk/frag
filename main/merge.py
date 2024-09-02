import json

# Load the first JSON file with UTF-8 encoding
with open(r'C:\Users\default.DESKTOP-7FKFEEG\frag\main\everything.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Load the second JSON file with UTF-8 encoding
with open(r'C:\Users\default.DESKTOP-7FKFEEG\frag\main\csv_output.json', 'r', encoding='utf-8') as f:
    wiki_parfum = json.load(f)

# Create a dictionary for quick lookup by name
wiki_parfum_dict = {entry['Name']: entry for entry in wiki_parfum}

# Define all possible attributes
attributes = [
    "Base notes", "Middle notes", "Top notes", "Notes", "Name", "Perfumer", 
    "Description", "Gender", "Concepts", "URL", "All reviews", 
    "Positive reviews", "Negative reviews", "Image URL", "Brand","Brand URL","Country","Perfume Name","Release Year","Perfume URL","Note"

]

# Merge datasets
merged_data = []

for entry in dataset:
    name = entry.get('Name')
    if name in wiki_parfum_dict:
        # Merge the entries
        merged_entry = {attr: entry.get(attr, "N/A") for attr in attributes}
        wiki_entry = wiki_parfum_dict[name]
        for attr in attributes:
            if attr in wiki_entry:
                merged_entry[attr] = wiki_entry[attr]
        merged_data.append(merged_entry)
    else:
        # Add the entry as is, ensuring all attributes are present
        merged_entry = {attr: entry.get(attr, "N/A") for attr in attributes}
        merged_data.append(merged_entry)

# Add remaining entries from wiki_parfum that were not in dataset
for name, entry in wiki_parfum_dict.items():
    if name not in [e['Name'] for e in dataset]:
        merged_entry = {attr: entry.get(attr, "N/A") for attr in attributes}
        merged_data.append(merged_entry)

# Save the merged data to a new JSON file
with open('everythingpt2.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=4)

print("Merging complete. The merged data is saved in 'merged_dataset.json'.")