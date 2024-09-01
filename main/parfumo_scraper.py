#STOPPED UPSERTING AT "https://basenotes.com/fragrances/orpheon-by-diptyque.26163070"
#URL #1847 OUT OF 23133 LEFT!!!!!  szx

import requests
from bs4 import BeautifulSoup
import numpy as np
import os
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import util
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Database connection
conn = psycopg2.connect(
    dbname="pgvector",
    user="postgres",
    password="postgres",
    host="localhost"
)
cur = conn.cursor()
def export_links_to_file(links, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for link in links:
                file.write(link + '\n')
        print(f"Successfully exported {len(links)} links to {file_path}")
    except Exception as e:
        print(f"An error occurred while exporting links: {e}")

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
CREATE TABLE IF NOT EXISTS wiki_parfum (
    id SERIAL PRIMARY KEY,
    notes TEXT,
    name TEXT,
    perfumer TEXT,
    description TEXT,
    gender TEXT,
    concepts TEXT,
    url TEXT UNIQUE
)
""")
conn.commit()
cleaned_lines = []
lines=[]
try:
    with open(r'C:\Users\default.DESKTOP-7FKFEEG\project\main\wiki_links.txt', encoding='utf-8') as f:
        lines = f.read().splitlines()
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()
for line in lines:
    if "https://www.wikiparfum.com/en/fragrances/" in line:
        if "?brand=" not in line:
            if "jpeg" and "png" and "jpg" and "img" and "svg" and "gif" and "ico" and "apng" and "pdf" and ".js" and "json" and "font" not in line:
                cleaned_lines.append(line)
export_links_to_file(cleaned_lines, r'C:\Users\default.DESKTOP-7FKFEEG\project\main\cleaned_wiki_links2.txt')
# WHERE TO PUT UPSERT FUNCTION!!!!!!!!
# Function to upsert data
def upsert_embedding(notes, name, perfumer, description, gender, concepts, url):
    try:
        # Convert numpy array to list
        
        # Debugging: Print the data being upserted
        print(f"Upserting data for URL: {url}")

        cur.execute("""
        INSERT INTO wiki_parfum (notes, name, perfumer, description, gender, concepts, url)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (url) DO UPDATE SET
            notes = EXCLUDED.notes,
            name = EXCLUDED.name,
            perfumer = EXCLUDED.perfumer,
            description = EXCLUDED.description,
            gender = EXCLUDED.gender,
            concepts = EXCLUDED.concepts
        """, (notes, name, perfumer, description, gender, concepts, url))
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
cleaned_lines=list(dict.fromkeys(cleaned_lines))
for line in cleaned_lines:
    if i <4000:
        i=i+1
        continue
    try:
        if "https://www.wikiparfum.com/en/fragrances/" in line:
            if "?brand=" not in line:
                if line != "https://www.wikiparfum.com/en/fragrances/":
                    print(f"Started on line #{i}\nURL: {line}")
                    try:
                        response = requests.get(line, timeout=10)  # Set a timeout of 10 seconds
                        response.raise_for_status()
                    except requests.exceptions.Timeout:
                        print(f"Timeout reached for URL: {line}. Skipping to the next line.")
                        continue
                    except requests.exceptions.RequestException as e:
                        print(f"Request failed for URL: {line}. Error: {e}")
                        continue
                    html_content = response.content.decode('utf-8')
                    soup = BeautifulSoup(html_content.replace("<br>", ""), "html.parser")
                    text_data = soup.get_text().strip()
                    text_data = text_data.replace("\n", "")
                    full_data = text_data
                    notes = name = perfumer = description = gender = concepts = ""
                    if "- Wikiparfum" in full_data:
                        name = full_data[0:full_data.index("- Wikiparfum")]
                        print(name)
                    if "olfactive classification" in full_data:
                        notes = full_data[full_data.index("all the ingredients") + 19:full_data.index("olfactive classification")]
                    if "by" in full_data:
                        perfumer = full_data[full_data.index("by") + 3:full_data.index("- Wikiparfum")]
                    if "PopularDescription" in full_data:
                        description = full_data[full_data.index("PopularDescription") + 18:full_data.index("Data sheetUser ratings")]
                    if "Concepts" in full_data:
                        concepts = full_data[full_data.index("Concepts") + 8:len(full_data)]
                    if "Gender" in full_data:
                        if full_data[full_data.index("Gender") + 6] == "M":
                            gender = "male"
                        elif full_data[full_data.index("Gender") + 6] == "F":
                            gender = "female"
                        elif full_data[full_data.index("Gender") + 6] == "U":
                            gender = "unisex"

                    data = {
                        "Notes": notes,
                        "Name": name,
                        "Perfumer": perfumer,
                        "description": description,
                        "gender": gender,
                        "concepts": concepts,
                        "URL": str(line)
                    }

                    to_append = {
                        "Notes": notes,
                        "Name": name,
                        "Perfumer": perfumer,
                        "description": description,
                        "gender": gender,
                        "concepts": concepts,
                        "URL": str(line)
                    }
                    total.append(to_append)

                    output_path = r'C:\Users\default.DESKTOP-7FKFEEG\project\main\wiki_parfum.json'
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(total, f, ensure_ascii=False, indent=4)
                    # Upsert into the database
                    upsert_embedding(notes, name,perfumer, description, gender, concepts, str(line))
                
            
    except ValueError as e:
        print(f"Error processing text data: {e}")
        continue
    i += 1
    print(f"Finished line with URL: {line}\nNumber completed: {i}\nNumber left: {len(cleaned_lines) - i}")


# Example usage
# Close the database connection
cur.close()
conn.close()
