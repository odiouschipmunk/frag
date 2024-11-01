import numpy as np
import psycopg2
from transformers import BertTokenizer, BertModel, pipeline, AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify, render_template
from sentence_transformers import util
import torch
import json
import asyncio
import aiohttp
import time
import ijson

app = Flask("im awesome")

# Load the locally trained model and tokenizer
model_path = r'C:\Users\default.DESKTOP-7FKFEEG\project\model2'
tokenizer_path = r'C:\Users\default.DESKTOP-7FKFEEG\project\tokenizer2'
etokenizer = BertTokenizer.from_pretrained(tokenizer_path)
emodel = BertModel.from_pretrained(model_path)

# Load the text generation model

try:

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct") 
    model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3.5-mini-instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 
    text_gen_pipeline=pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("Pipeline initialized successfully.")
except Exception as e:
    print(f"Error initializing pipeline: {e}")
    text_gen_pipeline = None

MAX_SEQUENCE_LENGTH = 131072

def truncate_sequence(sequence, max_length):
    return sequence[:max_length] if len(sequence) > max_length else sequence

def truncate_messages(messages, max_length):
    for message in messages:
        message['content'] = truncate_sequence(message['content'], max_length)
    return messages

messages = [
    {"role": "system", "content": "You are a fragrance bot that suggests the best fragrance to get based on user suggestions."},
]

def get_db_connection():
    try:
        return psycopg2.connect(
            dbname="pgvector",
            user="postgres",
            password="postgres",
            host="localhost"
        )
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")
        return None


def remove_unknown_urls():
    conn = get_db_connection()
    if conn is None:
        print("Failed to connect to the database.")
        return

    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM final WHERE url = 'Unknown' AND perfume_url = 'Unknown'")
            conn.commit()
            print("Records with 'Unknown' URLs have been removed.")
    except Exception as e:
        print(f"Error removing records: {e}")
    finally:
        conn.close()

remove_unknown_urls()

start_for_ai=time.time()
def rag(query, top_k=10, batch_size=1000):
    start_for_ai=time.time()
    conn = get_db_connection()
    if conn is None:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM final")
            total_records = cur.fetchone()[0]
            num_batches = (total_records // batch_size) + 1
            accumulated_results = []

            for batch_num in range(num_batches):
                offset = batch_num * batch_size
                cur.execute(f"SELECT * FROM final LIMIT {batch_size} OFFSET {offset}")
                results = cur.fetchall()
                
                # Log the structure of the results
                if not results:
                    print(f"Batch {batch_num} returned no results.")
                    continue

                base_notes = [result[1] for result in results]
                middle_notes = [result[2] for result in results]
                top_notes = [result[3] for result in results]
                notes = [result[4] for result in results]
                name = [result[5] for result in results]
                perfumer = [result[6] for result in results]
                description = [result[7] for result in results]
                gender = [result[8] for result in results]
                concepts = [result[9] for result in results]
                urls = [result[10] for result in results]
                all_reviews = [result[11] for result in results]
                positive_reviews = [result[12] for result in results]
                negative_reviews = [result[13] for result in results]
                review = [result[14] for result in results]
                image_url = [result[15] for result in results]
                brand = [result[16] for result in results]
                brand_url = [result[17] for result in results]
                country = [result[18] for result in results]
                perfume_name = [result[19] for result in results]
                release_year = [result[20] for result in results]
                perfume_url = [result[21] for result in results]
                note = [result[22] for result in results]

                # Filter out invalid embeddings
                valid_results = []
                for result in results:
                    embedding = result[23]
                    if isinstance(embedding, list) and all(isinstance(x, (float, int)) for sublist in embedding for x in sublist):
                        valid_results.append(result)
                    else:
                        print(f"Invalid embedding in result: {result}")

                if not valid_results:
                    print(f"Batch {batch_num} has no valid embeddings.")
                    continue

                embeddings_matrix = np.array([np.array(result[23]).flatten() for result in valid_results], dtype=np.float32)
                inputs = etokenizer(query, return_tensors='pt')
                with torch.no_grad():
                    outputs = emodel(**inputs)
                query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
                query_embedding = query_embedding.reshape(1, -1)
                embeddings_matrix = embeddings_matrix.reshape(len(valid_results), -1)
                scores = util.pytorch_cos_sim(query_embedding, embeddings_matrix)[0].cpu().numpy()
                batch_results = sorted(zip(base_notes, middle_notes, top_notes, notes, name, perfumer, description, gender, concepts, all_reviews, positive_reviews, negative_reviews, review, image_url, brand, brand_url, country, perfume_name, release_year, perfume_url, note, urls, scores), key=lambda x: x[22], reverse=True)
                accumulated_results.extend(batch_results)
            
            # Filter out null URLs and get top_k non-null results
            non_null_results = [
                result for result in accumulated_results 
                if (result[4] != "Unknown" and result[21] != "Unknown") or (result[19] != "Unknown" and result[17] != "Unknown") or (result[19] != "Unknown" and result[4] != "Unknown") or (result[21] != "Unknown" and result[17] != "Unknown")
            ]
            top_results = sorted(non_null_results, key=lambda x: x[22], reverse=True)[:top_k]

            # Return only the URL and the score for the top results, converting scores to float
            return top_results

    except Exception as e:
        print(f"Error querying embeddings: {e}")
        return []
    finally:
        conn.close()



async def generate_ai_answer(question):
    global messages
    start=time.time()
    messages.append({"role": "user", "content": question})
    messages = truncate_messages(messages, MAX_SEQUENCE_LENGTH)
    embed_answer = rag(question)
    embed_answer = [list(result) for result in embed_answer]
    for result in embed_answer:
        for i, r in enumerate(result):
            if type(r)==float:
                result[i]=str(r)
            if r is None or r == '':
                result[i] = "N/A"

    to_give=[]
    for result in embed_answer:
        if result != "N/A":
            to_give.append(result)
    # Extract relevant string data from embed_answer lists
    total = ""
    for result in to_give:
        try:
            total += str(result) + "\n"
        except Exception as e:
            print(f"Error extracting data: {e}")

    messages.append({"role": "system", "content": "Here is data that will help you answer the following question:\n\n" + total})
    # Add logging to track progress
    print("Generating AI answer...")
    try:
        # Clear GPU memory before inference
        torch.cuda.empty_cache()
        outputs=[]
        
        outputs = text_gen_pipeline(messages, max_new_tokens=1000)  # Reduce max_new_tokens
        print("AI answer generated.")

        # Clear GPU memory after inference
        torch.cuda.empty_cache()
        end_for_ai=time.time()
        print("Time taken to generate answer: ",end_for_ai-start_for_ai)
        print(outputs[0]['generated_text'][-1])
        return outputs[0]['generated_text'][-1]

    except Exception as e:
        print(f"Error generating AI answer: {e}")
        return "Error generating AI answer."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')
    if not query_text:
        return jsonify({'error': 'Query text is required'}), 400

    results = query_embeddings(query_text)
    return jsonify(results)

tasks = {}

async def background_task(task_id, question):
    # Simulate long processing time
    print(f"Starting background task {task_id} for question: {question}")
    answer = await generate_ai_answer(question)
    tasks[task_id] = answer
    print(f"Background task {task_id} completed.")

@app.route('/ai', methods=['POST'])
async def ai():
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'Question is required'}), 400

    task_id = str(len(tasks) + 1)
    tasks[task_id] = None
    asyncio.create_task(background_task(task_id, question))
    return jsonify({'task_id': task_id})

def query_embeddings(query, top_k=50, batch_size=1000):
    conn = get_db_connection()
    if conn is None:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM final")
            total_records = cur.fetchone()[0]
            num_batches = (total_records // batch_size) + 1

            accumulated_results = []

            for batch_num in range(num_batches):
                offset = batch_num * batch_size
                cur.execute(f"SELECT * FROM final LIMIT {batch_size} OFFSET {offset}")
                results = cur.fetchall()
                if not results:
                    continue

                urls = [result[10] for result in results]
                purl = [result[21] for result in results]
                embeddings_matrix = np.array([result[23] for result in results], dtype=np.float32)

                inputs = etokenizer(query, return_tensors='pt')
                with torch.no_grad():
                    outputs = emodel(**inputs)
                query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
                query_embedding = query_embedding.reshape(1, -1)
                embeddings_matrix = embeddings_matrix.reshape(len(results), -1)

                scores = util.pytorch_cos_sim(query_embedding, embeddings_matrix)[0].cpu().numpy()
                batch_results = sorted(zip(urls, purl, scores), key=lambda x: x[2], reverse=True)

                accumulated_results.extend(batch_results)

            # Filter out null URLs and get top_k non-null results
            non_null_results = [result for result in accumulated_results if result[0] is not None or result[1] is not None]
            top_results = sorted(non_null_results, key=lambda x: x[2], reverse=True)[:top_k]

            return [(result[0] if result[0] != "Unknown" else result[1] if result[1] != "Unknown" else "Not able to find link for this fragrance.", float(result[2])) for result in top_results]

    except Exception as e:
        print(f"Error querying embeddings: {e}")
        return []
    finally:
        conn.close()

@app.route('/ai_status/<task_id>', methods=['GET'])
async def ai_status(task_id):
    answer = tasks.get(task_id)
    if answer is None:
        return jsonify({'status': 'processing'})
    return jsonify({'status': 'complete', 'answer': answer})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)