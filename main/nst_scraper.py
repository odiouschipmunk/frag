#STOPPED UPSERTING AT "https://basenotes.com/fragrances/orpheon-by-diptyque.26163070"
#URL #1847 OUT OF 23133 LEFT!!!!!  szx

import requests
from bs4 import BeautifulSoup
import json


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

cleaned_lines = []
lines=[]
try:
    with open(r'C:\Users\default.DESKTOP-7FKFEEG\project\main\nst.txt', encoding='utf-8') as f:
        lines = f.read().splitlines()
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()
for line in lines:
    if "https://nstperfume.com" in line:
        if "jpeg" and "png" and "jpg" and "img" and "svg" and "gif" and "ico" and "apng" and "pdf" and ".js" and "json" and "font" not in line:
            cleaned_lines.append(line)
export_links_to_file(cleaned_lines, r'C:\Users\default.DESKTOP-7FKFEEG\project\main\cleaned_nst.txt')
# WHERE TO PUT UPSERT FUNCTION!!!!!!!!
# Function to upsert data

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
    i=i+1
    print(i)
    try:
        if "https://nstperfume.com/" in line:
            years = [str(year) for year in range(1990, 2025)]
            if not any(year in line for year in years):
                continue
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
            name=""
            review=""
            if "fragrance review" in full_data:
                review = text_data[text_data.index("fragrance review"):-1]
                name= text_data[0:text_data.index("fragrance review")]
            data = {
                "review":review,
                "name": name
            }

            to_append = {
                "review":review,
                "name": name
            }
            total.append(to_append)

            output_path = r'C:\Users\default.DESKTOP-7FKFEEG\project\main\nst.json'
            if review != "":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(total, f, ensure_ascii=False, indent=4)
                    # Upsert into the database
                
            
    except ValueError as e:
        print(f"Error processing text data: {e}")
        continue
    i += 1
    print(f"Finished line with URL: {line}\nNumber completed: {i}\nNumber left: {len(cleaned_lines) - i}")


