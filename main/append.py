import json

# Load merge.json with UTF-8 encoding
with open('merge.json', 'r', encoding='utf-8') as f:
    merge_data = json.load(f)

# Load nst.json with UTF-8 encoding
with open('nst.json', 'r', encoding='utf-8') as f:
    nst_data = json.load(f)

# Append nst_data to merge_data
merge_data.extend(nst_data)

# Save the combined data back to merge.json
with open('merge.json', 'w', encoding='utf-8') as f:
    json.dump(merge_data, f, indent=4)

print("Appending complete. The combined data is saved in 'merge.json'.")