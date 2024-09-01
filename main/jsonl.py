import json
with open('combined.json', 'r', encoding='utf-8') as f:
    JSON_file = json.load(f)
with open('output.jsonl', 'w') as outfile:
    for entry in JSON_file:
        json.dump(entry, outfile)
        outfile.write('\n')