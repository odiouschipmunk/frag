'''
lines=[]
with open('C:\\Users\\default.DESKTOP-7FKFEEG\\project\\frag\\scraper\\urls.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()

from selenium import webdriver
from nium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import re
import time
from bs4 import BeautifulSoup

# Setup Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument('--ignore-certificate-errors')
chrome_options.add_argument('--ignore-ssl-errors')

# Automatically manage the ChromeDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)     # Replace multiple spaces with a single space
    text = text.strip()                  # Remove leading/trailing spaces
    return text

# Access the website
i = 0
for line in lines:
    if ".jpg" in line:
        continue
    for retry in range(3):  # Retry mechanism
        try:
            driver.get(line)
            break
        except Exception as e:
            print(f"Attempt {retry + 1} failed: {e}")
            time.sleep(2)
    print(f"Started on line #{i}\nURL: {line}")
    content = driver.page_source
    soup = BeautifulSoup(content, "html.parser")
    text_data = soup.get_text()
    text_data = " ".join([p.get_text() for p in text_data])
    
    with open("website_data.txt", "a", encoding="utf-8") as file:
        file.write(text_data)

    cleaned_text = preprocess_text(text_data)
    with open("cleaned.txt", "a", encoding="utf-8") as file:
        file.write(cleaned_text)
    i += 1
    print(f"Finished line with URL: {line}\nNumber completed: {i}")

driver.quit()
'''

import requests
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="f62f03a2-edd7-4175-b97f-3bc401131a68")
index=pc.Index("frag")
lines = []
with open('C:\\Users\\default.DESKTOP-7FKFEEG\\project\\frag\\scraper\\urls.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()
# Loop through each URL
i = 0
pages=[[]]
for line in lines:

    if "fragrance" or "review" or "community" not in line:
        continue
    print(f"Started on line #{i}\nURL: {line}")
    response=requests.get(line)
    html_content=response.content.decode('utf-8')
    soup=BeautifulSoup(html_content.replace("<br>",""), "html.parser")
    text_data=soup.get_text().strip()
    print(text_data)

    embedding=model.encode("Website: "+str(text_data)+"\n\nLink: "+str(line))
    index.upsert(
        vectors=[
            {"id":str(i),
             "values":embedding,
             "metadata":{
                 "page":str(text_data),
                 "link":str(line),
             }}
        ]
    )
    pages.append([text_data, line])
    with open("website_data.txt",'a',encoding='utf-8') as f:
        f.write(str(pages[i][0])+"\n"+pages[i][1]+"\n\n\n")
    i += 1
    print(f"Finished line with URL: {line}\nNumber completed: {i}")

