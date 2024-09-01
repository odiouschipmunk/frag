#STOPPED UPSERTING AT "https://basenotes.com/fragrances/orpheon-by-diptyque.26163070"
#URL #1847 OUT OF 23133 LEFT!!!!!  szx

import requests
from bs4 import BeautifulSoup
import numpy as np
import os
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import util
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
torch.cuda.empty_cache()
model_name = "deepset/roberta-base-squad2"

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Use a smaller model or offload parts to CPU

model = T5ForConditionalGeneration.from_pretrained(model_name)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Database connection
conn = psycopg2.connect(
    dbname="pgvector",
    user="postgres",
    password="postgres",
    host="localhost"
)
cur = conn.cursor()

def split_string(input_string, max_length=511):
    """
    Splits the input string into pieces, each less than max_length characters.
    
    Args:
    input_string (str): The string to be split.
    max_length (int): The maximum length of each piece. Default is 511.
    
    Returns:
    List[str]: A list of string pieces.
    """
    # Ensure max_length is at least 1 to avoid infinite loop
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    
    # Split the string into pieces
    pieces = [input_string[i:i + max_length] for i in range(0, len(input_string), max_length)]
    
    return pieces
# Create table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS bfrag (
    id SERIAL PRIMARY KEY,
    base TEXT,
    middle TEXT,
    top TEXT,
    notes TEXT,
    name TEXT,
    reviews TEXT,
    previews TEXT,
    nreviews TEXT,
    url TEXT UNIQUE,
    embedding FLOAT8[]
)
""")
conn.commit()

lines = []
try:
    with open(r'C:\Users\default.DESKTOP-7FKFEEG\project\main\crawled urls.txt', encoding='utf-8') as f:
        lines = f.read().splitlines()
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

# Filter out URLs containing image extensions
cleaned_lines = [line for line in lines if "https://basenotes.com/fragrances" in line and not any(ext in line for ext in ["jpeg", "png", "jpg", "img", "svg", "gif", "ico", "apng", "pdf", ".js", "json", "font"])]
'''
def trunc(input_string, max_bytes=39500, encoding='utf-8'):
    encoded_string = input_string.encode(encoding)
    if len(encoded_string) <= max_bytes:
        return input_string
    truncated_string = encoded_string[:max_bytes]
    return truncated_string.decode(encoding, errors='ignore')
'''
# WHERE TO PUT UPSERT FUNCTION!!!!!!!!
# Function to upsert data
def upsert_embedding(base, middle, top, notes, name, reviews, previews, nreviews, url, embedding):
    try:
        # Convert numpy array to list
        embedding_list = embedding.tolist()
        
        # Debugging: Print the data being upserted
        print(f"Upserting data for URL: {url}")

        cur.execute("""
        INSERT INTO bfrag (base, middle, top, notes, name, reviews, previews, nreviews, url, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (url) DO UPDATE SET
            base = EXCLUDED.base,
            middle = EXCLUDED.middle,
            top = EXCLUDED.top,
            notes = EXCLUDED.notes,
            name = EXCLUDED.name,
            reviews = EXCLUDED.reviews,
            previews = EXCLUDED.previews,
            nreviews = EXCLUDED.nreviews,
            embedding = EXCLUDED.embedding
        """, (base, middle, top, notes, name, reviews, previews, nreviews, url, embedding_list))
        conn.commit()
        print(f"Upserted record for URL: {url}")
    except Exception as e:
        conn.rollback()
        print(f"Error upserting data to the database: {e}")

# Function to process data in batches
def process_in_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Loop through cleaned URLs and text content
i = 0
pages = []
total = []
batch_size = 1  # Set batch size to 1 for smaller batches
for line in cleaned_lines:
    print(f"Started on line #{i}\nURL: {line}")
    response = requests.get(line, timeout=10)  # Set a timeout of 10 seconds
    response.raise_for_status()
    html_content = response.content.decode('utf-8')
    soup = BeautifulSoup(html_content.replace("<br>", ""), "html.parser")
    text_data = soup.get_text().strip()
    text_data = text_data.replace("\n", "")
    full_data= text_data
    reviews=""
    previews=""
    nreviews=""
    base=middle=top=notes=""

    if "reviews" in line:
        name= full_data[text_data.index("Reviews of "):text_data.index("– Basenotes")]
        if "positive" in line:
            try:
                previews=text_data[text_data.index("Positive Reviews of"):text_data.index("Loving Perfume on the Internet since 2000")]
            except ValueError as e:
                print(f"Error processing reviews: {e}")

        elif "negative" in line:
            try:
                nreviews=text_data[text_data.index("Negative Reviews of"):text_data.index("Loving Perfume on the Internet since 2000")]
            except ValueError as e:
                print(f"Error processing reviews: {e}")

        elif "neutral" in line:
            try:
                reviews=text_data[text_data.index("Neutral Reviews of"):text_data.index("Loving Perfume on the Internet since 2000")]
            except ValueError as e:
                print(f"Error processing reviews: {e}")

        else:
            reviews=text_data[text_data.index("Show all reviews"):text_data.index("Loving Perfume on the Internet since 2000")]
        to_encode = notes + "\n\nText data: " + text_data
        input_ids = tokenizer(to_encode, return_tensors="pt", truncation=True).input_ids.to(device)

        # Process in smaller batches
        for batch in process_in_batches(input_ids, batch_size):
            outputs = model.generate(batch, max_new_tokens=50)  # Set max_new_tokens to control generation length
            embedding = outputs[0].cpu().numpy()  # Move tensor to CPU and convert to numpy array

            data = {
                "Base notes": base,
                "Middle notes": middle,
                "Top notes": top,
                "Notes": notes,
                "Name": name,
                "All reviews": reviews,
                "Positive reviews": previews,
                "Negative reviews": nreviews,
                "URL": str(line),
                "Embedding": embedding.tolist()  # Convert numpy array to list for JSON serialization
            }
            to_append = {
                "Base notes": base,
                "Middle notes": middle,
                "Top notes": top,
                "Notes": notes,
                "Name": name,
                "All reviews": reviews,
                "Positive reviews": previews,
                "Negative reviews": nreviews,
                "URL": str(line)
            }
            total.append(to_append)

            output_path = r'C:\Users\default.DESKTOP-7FKFEEG\project\main\bigdataset.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(total, f, ensure_ascii=False, indent=4)
        # Upsert into the database
        upsert_embedding(base, middle, top, notes, name, reviews, previews, nreviews,str(line), embedding)
        continue

    try:
        if "reviews" in line:
            continue
        if "fragrance notesHead" in text_data:
            str1 = "fragrance notesHead"
            str2 = "Latest Reviews"
            idx1 = text_data.index(str1)
            idx2 = text_data.index(str2)
            text_data = text_data[idx1+len(str1):idx2]
            top = text_data[0:text_data.index("Heart")]
            middle = text_data[text_data.index("Heart")+len("Heart"):text_data.index("Base")]
            base = text_data[text_data.index("Heart")+len(middle)+9:]
            if "– Basenotes" in text_data:
                name= full_data[0:full_data.index("– Basenotes")]
                notes = f"top notes: {top}, middle notes: {middle}, base notes: {base}"
            else:
                name= full_data[0:full_data.index("Where to buy")]
                notes = f"top notes: {top}, middle notes: {middle}, base notes: {base}"
        else:
            str1 = 'fragrance notes'
            str2 = "Latest Reviews"
            idx1 = text_data.index(str1)
            idx2 = text_data.index(str2)
            text_data = text_data[idx1+len(str1):idx2]
            notes = text_data
            top = middle = base = " "
            if "– Basenotes" in text_data:
                name= full_data[0:full_data.index("– Basenotes")]
                notes = f"top notes: {top}, middle notes: {middle}, base notes: {base}"
            else:
                name= full_data[0:full_data.index("Where to buy")]
                notes = f"top notes: {top}, middle notes: {middle}, base notes: {base}"

                    
        to_encode = notes + "\n\nText data: " + text_data
        input_ids = tokenizer(to_encode, return_tensors="pt", truncation=True).input_ids.to(device)

        # Process in smaller batches
        for batch in process_in_batches(input_ids, batch_size):
            outputs = model.generate(batch, max_new_tokens=50)  # Set max_new_tokens to control generation length
            embedding = outputs[0].cpu().numpy()  # Move tensor to CPU and convert to numpy array

            data = {
                "Base notes": base,
                "Middle notes": middle,
                "Top notes": top,
                "Notes": notes,
                "Name": name,
                "All reviews": reviews,
                "Positive reviews": previews,
                "Negative reviews": nreviews,
                "URL": str(line),
                "Embedding": embedding.tolist()  # Convert numpy array to list for JSON serialization
            }
            to_append = {
                "Base notes": base,
                "Middle notes": middle,
                "Top notes": top,
                "Notes": notes,
                "Name": name,
                "All reviews": reviews,
                "Positive reviews": previews,
                "Negative reviews": nreviews,
                "URL": str(line)
            }
            total.append(to_append)

            # Upsert into the database
            upsert_embedding(base, middle, top, notes, name, reviews, previews, nreviews,str(line), embedding)
        output_path = r'C:\Users\default.DESKTOP-7FKFEEG\project\main\bigdataset.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(total, f, ensure_ascii=False, indent=4)
    except ValueError as e:
        print(f"Error processing text data: {e}")
        continue

    output_path = r'C:\Users\default.DESKTOP-7FKFEEG\project\main\bigdataset.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(total, f, ensure_ascii=False, indent=4)
    
    i += 1
    print(f"Finished line with URL: {line}\nNumber completed: {i}\nNumber left: {len(cleaned_lines) - i}")

output_path = r'C:\Users\default.DESKTOP-7FKFEEG\project\main\everything.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(total, f, ensure_ascii=False, indent=4)
def query_embeddings(query, top_k=5):
    cur.execute("SELECT url, embedding FROM bfrag")
    embeddings = cur.fetchall()
    urls = [embedding[0] for embedding in embeddings]
    embeddings_matrix = np.array([embedding[1] for embedding in embeddings])
    query_embedding = model.encode(query, clean_up_tokenization_spaces=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings_matrix)[0].cpu().numpy()
    sorted_results = sorted(zip(urls, scores), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]
def query(q):
    input_ids = tokenizer(q, return_tensors="pt", truncation=True).input_ids.to("cuda")
    outputs = model.generate(input_ids, max_length=4096)
    return tokenizer.decode(outputs[0])
# Example usage
print(query("What are some good vanilla fragrances for men?"))
# Close the database connection
cur.close()
conn.close()
