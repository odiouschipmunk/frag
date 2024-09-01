import json
import torch
from transformers import BertTokenizer, BertModel
import psycopg2
from psycopg2.extras import execute_values

# Load the JSON data


# Load the embedding model and tokenizer
model_path = "C:/Users/default.DESKTOP-7FKFEEG/project/model2"
tokenizer_path = "C:/Users/default.DESKTOP-7FKFEEG/project/tokenizer2"
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertModel.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with open('combined.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
# Function to generate embeddings
def generate_embedding(text, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    dbname="pgvector",
    user="postgres",
    password="postgres",
    host="localhost"
)
cur = conn.cursor()

# Create the "rag" table if it doesn't exist
cur.execute("""
CREATE TABLE IF NOT EXISTS rag (
    id SERIAL PRIMARY KEY,
    base_notes TEXT,
    middle_notes TEXT,
    top_notes TEXT,
    notes TEXT,
    name TEXT,
    perfumer TEXT,
    description TEXT,
    gender TEXT,
    concepts TEXT,
    url TEXT UNIQUE,
    all_reviews TEXT,
    positive_reviews TEXT,
    negative_reviews TEXT,
    review TEXT,
    embedding FLOAT8[]
)
""")
conn.commit()

# Add unique constraint to the "url" column if it doesn't exist
cur.execute("""
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name='rag' AND constraint_type='UNIQUE' AND constraint_name='unique_url'
    ) THEN
        ALTER TABLE rag ADD CONSTRAINT unique_url UNIQUE (url);
    END IF;
END $$;
""")
conn.commit()

# Function to upsert data
def upsert_embedding(base_notes, middle_notes, top_notes, notes, name, perfumer, description, gender, concepts, url, all_reviews, positive_reviews, negative_reviews, review, embedding):
    try:
        embedding_list = embedding.tolist()
        cur.execute("""
        INSERT INTO rag (base_notes, middle_notes, top_notes, notes, name, perfumer, description, gender, concepts, url, all_reviews, positive_reviews, negative_reviews, review, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (url) DO UPDATE SET
            base_notes = EXCLUDED.base_notes,
            middle_notes = EXCLUDED.middle_notes,
            top_notes = EXCLUDED.top_notes,
            notes = EXCLUDED.notes,
            name = EXCLUDED.name,
            perfumer = EXCLUDED.perfumer,
            description = EXCLUDED.description,
            gender = EXCLUDED.gender,
            concepts = EXCLUDED.concepts,
            all_reviews = EXCLUDED.all_reviews,
            positive_reviews = EXCLUDED.positive_reviews,
            negative_reviews = EXCLUDED.negative_reviews,
            review = EXCLUDED.review,
            embedding = EXCLUDED.embedding
        """, (base_notes, middle_notes, top_notes, notes, name, perfumer, description, gender, concepts, url, all_reviews, positive_reviews, negative_reviews, review, embedding_list))
        conn.commit()
        print(f"Upserted record for URL: {url}")
    except Exception as e:
        conn.rollback()
        print(f"Error upserting data to the database: {e}")

# Upsert data into the "rag" table
for i, item in enumerate(data):
    try:
        text = json.dumps(item)  # Convert the JSON item to a string
        embedding = generate_embedding(text)
        upsert_embedding(
            item.get("Base notes", ""),
            item.get("Middle notes", ""),
            item.get("Top notes", ""),
            item.get("Notes", ""),
            item.get("Name", ""),
            item.get("Perfumer", ""),
            item.get("Description", ""),
            item.get("Gender", ""),
            item.get("Concepts", ""),
            item.get("URL", ""),
            item.get("All reviews", ""),
            item.get("Positive reviews", ""),
            item.get("Negative reviews", ""),
            item.get("review", ""),
            embedding
        )
        print(f"Inserted item {i+1}/{len(data)}")
    except Exception as e:
        print(f"Error inserting item {i+1}: {e}")

# Close the database connection
cur.close()
conn.close()

print("Data upserted and embedded successfully.")