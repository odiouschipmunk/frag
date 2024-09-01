import requests
from transformers import AutoModel
from bs4 import BeautifulSoup
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, util

# Initialize Pinecone
pc=Pinecone(api_key="f62f03a2-edd7-4175-b97f-3bc401131a68")
index = pc.Index("self_crawled")

# Initialize Sentence Transformer model
model = SentenceTransformer('Lajavaness/bilingual-embedding-large', trust_remote_code=True)
# Read URLs from the file
lines = []
with open(r'C:\Users\default.DESKTOP-7FKFEEG\project\frag\scraper\cleaned_urls.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()

# Filter out URLs containing image extensions
cleaned_lines = []
for line in lines:
    if not any(ext in line for ext in ["jpeg", "png", "jpg", "img", "svg", "gif", "ico", "apng", "pdf", ".js", "json", "font"]):
        cleaned_lines.append(line)

# Write cleaned URLs to a new file
#with open(r'C:\Users\default.DESKTOP-7FKFEEG\project\frag\scraper\cleaned_urls.txt', 'w', encoding='utf-8') as f:
    #f.write('\n'.join(cleaned_lines))
def trunc(input_string, max_bytes=39500, encoding='utf-8'):
    encoded_string = input_string.encode(encoding)
    
    if len(encoded_string) <= max_bytes:
        return input_string
    truncated_string = encoded_string[:max_bytes]
    return truncated_string.decode(encoding, errors='ignore')

# Loop through cleaned URLs
i = 0
pages = []
'''
for line in cleaned_lines:
    print(f"Started on line #{i}\nURL: {line}")
    
    response = requests.get(line)
    html_content = response.content.decode('utf-8')
    soup = BeautifulSoup(html_content.replace("<br>", ""), "html.parser")
    text_data = soup.get_text().strip()
    text_data=trunc(text_data.replace("\n",""),max_bytes=39500,encoding='utf-8')
    line=trunc(line,max_bytes=39500,encoding='utf-8')
    # Generate embedding with clean_up_tokenization_spaces set explicitly
    embedding = model.encode("Website: " + str(text_data) + "\n\nLink: " + str(line))

    # Upsert data into Pinecone
    index.upsert( 
        vectors=[
            {
                "id": str(i),
                "values": embedding.tolist(),  # Ensure embedding is converted to a list
                "metadata": {
                    "page": str(text_data),
                    "link": str(line), 
                }
            }
        ]
    )

    # Save data to pages list and file
    pages.append([text_data, line])
    with open("website_data.txt", 'a', encoding='utf-8') as f:
        f.write(str(pages[i]) + "\n\n\n")

    i += 1
    print(f"Finished line with URL: {line}\nNumber completed: {i}")
'''
def query(q):
    result=index.query(
        vector=model.encode(q).tolist(),
        top_k=5,
        include_metadata=True
    )
    return result
print(query("a vanilla, tobacco fragrance for men"))
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
    if "fragrances" or "reviews" or "community" not in line:
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

'''