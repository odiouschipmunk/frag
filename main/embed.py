import os
import json
import torch
import numpy as np
import psycopg2
from concurrent.futures import ThreadPoolExecutor, as_completed
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer, util

os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
torch.cuda.empty_cache()

# Load the pre-trained mxbai-embed-large-v1 model
model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', truncate_dim=4096)

# Database connection
def get_db_connection():
    return psycopg2.connect(
        dbname="pgvector",
        user="postgres",
        password="postgres",
        host="localhost"
    )

def create_table():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
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

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        exit()

def upsert_embedding(base, middle, top, notes, name, reviews, previews, nreviews, url, embedding):
    try:
        embedding_list = embedding.tolist()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
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
        print(f"Error upserting data to the database: {e}")

def process_data(data):
    try:
        base = data.get('Base notes', '')
        middle = data.get('Middle notes', '')
        top = data.get('Top notes', '')
        notes = data.get('Notes', '')
        name = data.get('Name', '')
        reviews = data.get('All reviews', '')
        previews = data.get('Positive reviews', '')
        nreviews = data.get('Negative reviews', '')
        url = data.get('URL', '')
        text_data = notes + "\n\nText data: " + data.get('Text data', '')

        to_encode = notes + "\n\nText data: " + text_data
        embedding = model.encode(to_encode)

        upsert_embedding(base, middle, top, notes, name, reviews, previews, nreviews, url, embedding)
    except Exception as e:
        print(f"Error processing data: {e}")

def main():
    create_table()
    data_list = load_json(r'C:\Users\default.DESKTOP-7FKFEEG\project\main\dataset.json')
    '''
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_data = {executor.submit(process_data, data): data for data in data_list}
        for future in as_completed(future_to_data):
            future.result()
    '''

if __name__ == "__main__":
    main()

def query_embeddings(query, top_k=5):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT url, embedding FROM bfrag")
            embeddings = cur.fetchall()
            urls = [embedding[0] for embedding in embeddings]
            embeddings_matrix = np.array([embedding[1] for embedding in embeddings], dtype=np.float32)
            query_embedding = model.encode(query).astype(np.float32)
            scores = util.pytorch_cos_sim(query_embedding, embeddings_matrix)[0].cpu().numpy()
            sorted_results = sorted(zip(urls, scores), key=lambda x: x[1], reverse=True)
            return sorted_results[:top_k]

def query(q):
    return query_embeddings(q)

print(query("pineapple, vanilla, fragrances for men"))