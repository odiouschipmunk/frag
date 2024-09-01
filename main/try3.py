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

app = Flask(__name__)

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

def rag(query, top_k=10):
    conn = get_db_connection()
    if conn is None:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM rag")
            results = cur.fetchall()
            if not results:
                return []

            # Assuming the first column is the URL and the second column is the embedding
            urls = [result[10] for result in results]
            base_notes = [result[1] for result in results]
            middle_notes = [result[2] for result in results]
            top_notes = [result[3] for result in results]
            notes = [result[4] for result in results]
            name = [result[5] for result in results]
            description = [result[7] for result in results]
            concepts = [result[9] for result in results]
            all_reviews = [result[11] for result in results]
            review = [result[14] for result in results]
            embeddings_matrix = np.array([result[15] for result in results], dtype=np.float32)

            inputs = etokenizer(query, return_tensors='pt')
            with torch.no_grad():
                outputs = emodel(**inputs)
            query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
            query_embedding = query_embedding.reshape(1, -1)
            embeddings_matrix = embeddings_matrix.reshape(len(results), -1)

            scores = util.pytorch_cos_sim(query_embedding, embeddings_matrix)[0].cpu().numpy()
            sorted_results = sorted(zip(base_notes, middle_notes, top_notes, notes, name, description, concepts, all_reviews, review, urls, scores), key=lambda x: x[10], reverse=True)

            # Filter out null URLs and get top_k non-null results
            non_null_results = [result for result in sorted_results if result[9] and result[5] is not None]
            top_results = non_null_results[:top_k]

            # Return only the URL and the score for the top results, converting scores to float
            return top_results
    except Exception as e:
        print(f"Error querying embeddings: {e}")
        return []
    finally:
        conn.close()
async def generate_ai_answer(question):
    global messages
    messages.append({"role": "user", "content": question})
    messages = truncate_messages(messages, MAX_SEQUENCE_LENGTH)
    embed_answer = rag(question)
    embed_answer = [list(result) for result in embed_answer]
    for result in embed_answer:
        for i, r in enumerate(result):
            if r is None or r == '':
                result[i] = "N/A"
        result[10] = float(result[10])
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
        print(type(text_gen_pipeline))
        print(type(outputs))
        
        outputs = text_gen_pipeline(messages, max_new_tokens=1000)  # Reduce max_new_tokens
        print("AI answer generated.")

        # Clear GPU memory after inference
        torch.cuda.empty_cache()
        history=outputs[0]['generated_text'][-1]
        to_return=history
        outputs[0]['generated_text']=outputs[0]['generated_text'][:-2]
        history=json.dumps(history)
        messages.append({"role": "system", "content": "Here is what you have written in the past: "+history})
        return to_return

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

def query_embeddings(query, top_k=50):
    conn = get_db_connection()
    if conn is None:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM rag")
            results = cur.fetchall()
            if not results:
                return []

            # Assuming the first column is the URL and the second column is the embedding
            urls = [result[10] for result in results]
            embeddings_matrix = np.array([result[15] for result in results], dtype=np.float32)

            inputs = etokenizer(query, return_tensors='pt')
            with torch.no_grad():
                outputs = emodel(**inputs)
            query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
            query_embedding = query_embedding.reshape(1, -1)
            embeddings_matrix = embeddings_matrix.reshape(len(results), -1)

            scores = util.pytorch_cos_sim(query_embedding, embeddings_matrix)[0].cpu().numpy()
            sorted_results = sorted(zip(urls, scores), key=lambda x: x[1], reverse=True)

            # Filter out null URLs and get top_k non-null results
            non_null_results = [result for result in sorted_results if result[0] is not None]
            top_results = non_null_results[:top_k]

            # If there are not enough non-null results, continue searching
            if len(top_results) < top_k:
                additional_results = sorted_results[len(non_null_results):]
                for result in additional_results:
                    if result[0] is not None:
                        top_results.append(result)
                        if len(top_results) == top_k:
                            break

            # Return only the URL and the score for the top results, converting scores to float
            return [(result[0], float(result[1])) for result in top_results]
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