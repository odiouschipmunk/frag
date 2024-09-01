import requests
from bs4 import BeautifulSoup
import numpy as np
import os
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import util
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import json
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
torch.cuda.empty_cache()

model_name = 'distilbert-base-uncased-distilled-squad'

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
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
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    return [input_string[i:i + max_length] for i in range(0, len(input_string), max_length)]

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

cleaned_lines = [line for line in lines if "https://basenotes.com/fragrances" in line and not any(ext in line for ext in ["jpeg", "png", "jpg", "img", "svg", "gif", "ico", "apng", "pdf", ".js", "json", "font"])]

def upsert_embedding(base, middle, top, notes, name, reviews, previews, nreviews, url, embedding):
    try:
        embedding_list = embedding.tolist()
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
    except Exception as e:
        conn.rollback()
        print(f"Error upserting data to the database: {e}")

def process_in_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def process_url(line):
    try:
        response = requests.get(line, timeout=10)
        response.raise_for_status()
        html_content = response.content.decode('utf-8')
        soup = BeautifulSoup(html_content.replace("<br>", ""), "html.parser")
        text_data = soup.get_text().strip().replace("\n", "")
        full_data = text_data
        reviews = previews = nreviews = base = middle = top = notes = ""

        if "reviews" in line:
            name = full_data[text_data.index("Reviews of "):text_data.index("– Basenotes")]
            if "positive" in line:
                try:
                    previews = text_data[text_data.index("Positive Reviews of"):text_data.index("Loving Perfume on the Internet since 2000")]
                except ValueError as e:
                    print(f"Error processing reviews: {e}")

            elif "negative" in line:
                try:
                    nreviews = text_data[text_data.index("Negative Reviews of"):text_data.index("Loving Perfume on the Internet since 2000")]
                except ValueError as e:
                    print(f"Error processing reviews: {e}")

            elif "neutral" in line:
                try:
                    reviews = text_data[text_data.index("Neutral Reviews of"):text_data.index("Loving Perfume on the Internet since 2000")]
                except ValueError as e:
                    print(f"Error processing reviews: {e}")

            else:
                reviews = text_data[text_data.index("Show all reviews"):text_data.index("Loving Perfume on the Internet since 2000")]

        else:
            if "fragrance notesHead" in text_data:
                str1 = "fragrance notesHead"
                str2 = "Latest Reviews"
                idx1 = text_data.index(str1)
                idx2 = text_data.index(str2)
                text_data = text_data[idx1+len(str1):idx2]
                top = text_data[0:text_data.index("Heart")]
                middle = text_data[text_data.index("Heart")+len("Heart"):text_data.index("Base")]
                base = text_data[text_data.index("Heart")+len(middle)+9:]
                name = full_data[0:full_data.index("– Basenotes")]
                notes = f"top notes: {top}, middle notes: {middle}, base notes: {base}"
            else:
                str1 = 'fragrance notes'
                str2 = "Latest Reviews"
                idx1 = text_data.index(str1)
                idx2 = text_data.index(str2)
                text_data = text_data[idx1+len(str1):idx2]
                notes = text_data
                top = middle = base = " "
                name = full_data[0:full_data.index("– Basenotes")]

        return (base, middle, top, notes, name, reviews, previews, nreviews, line, text_data)
    except Exception as e:
        print(f"Error processing URL {line}: {e}")
        return None

def main():
    total = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(process_url, line): line for line in cleaned_lines}
        for future in as_completed(future_to_url):
            result = future.result()
            if result:
                base, middle, top, notes, name, reviews, previews, nreviews, line, text_data = result

                base_pieces = split_string(base)
                middle_pieces = split_string(middle)
                top_pieces = split_string(top)
                notes_pieces = split_string(notes)
                name_pieces = split_string(name)
                reviews_pieces = split_string(reviews)
                previews_pieces = split_string(previews)
                nreviews_pieces = split_string(nreviews)

                for base_piece, middle_piece, top_piece, notes_piece, name_piece, reviews_piece, previews_piece, nreviews_piece in zip(base_pieces, middle_pieces, top_pieces, notes_pieces, name_pieces, reviews_pieces, previews_pieces, nreviews_pieces):
                    to_encode = notes_piece + "\n\nText data: " + text_data
                    input_ids = tokenizer(to_encode, return_tensors="pt", truncation=True).input_ids.to(device)

                    for batch in process_in_batches(input_ids, batch_size=8):
                        outputs = model.generate(batch, max_new_tokens=50)
                        embedding = outputs[0].cpu().numpy()

                        data = {
                            "Base notes": base_piece,
                            "Middle notes": middle_piece,
                            "Top notes": top_piece,
                            "Notes": notes_piece,
                            "Name": name_piece,
                            "All reviews": reviews_piece,
                            "Positive reviews": previews_piece,
                            "Negative reviews": nreviews_piece,
                            "URL": str(line),
                            "Embedding": embedding.tolist()
                        }
                        to_append = {
                            "Base notes": base_piece,
                            "Middle notes": middle_piece,
                            "Top notes": top_piece,
                            "Notes": notes_piece,
                            "Name": name_piece,
                            "All reviews": reviews_piece,
                            "Positive reviews": previews_piece,
                            "Negative reviews": nreviews_piece,
                            "URL": str(line)
                        }
                        total.append(to_append)

                        upsert_embedding(base_piece, middle_piece, top_piece, notes_piece, name_piece, reviews_piece, previews_piece, nreviews_piece, str(line), embedding)

        output_path = r'C:\Users\default.DESKTOP-7FKFEEG\project\main\bigdataset.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(total, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()

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

print(query("What are some good vanilla fragrances for men?"))

cur.close()
conn.close()