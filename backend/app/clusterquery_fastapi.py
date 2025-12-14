import json
import os
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException
from groq import AsyncGroq  # Use the Asynchronous Groq client
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
from fuzzywuzzy import process, fuzz
from dotenv import load_dotenv


# Load .env file from the parent directory
from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")

# Read credentials
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_FILE_PATH = Path(__file__).parent.parent / "data" / "data.jsonl"
MAX_ITEMS_PER_CLUSTER = 5
MAX_TEXT_LENGTH = 200
MAX_RECORDS_PER_CLUSTERING = 2000

# --- Global Clients & Models (Initialized lazily to handle missing API keys) ---
groq_client = None
embedding_model = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI-Powered Data Clustering API",
    description="An API to load text data from JSON files, cluster it based on semantic meaning, and label the clusters using an LLM.",
    version="1.2.1"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class ChatRequest(BaseModel):
    question: str
    cluster_name: str = None
    cluster_data: List[Dict[str, Any]] = []

# --- Global variable to store clustered data ---
clustered_data_cache = None
cluster_names_cache = []

# --- Client Initialization Functions (Lazy Loading) ---
def get_groq_client():
    """Lazily initialize and return the Groq AsyncGroq client."""
    global groq_client
    if groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set. Please set the environment variable or add it to .env file.")
        try:
            groq_client = AsyncGroq(api_key=GROQ_API_KEY)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq client: {e}")
    return groq_client

def get_embedding_model():
    """Lazily initialize and return the embedding model."""
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {e}")
    return embedding_model

# --- Helper Functions ---
def read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Reads and parses a JSON file, returning all records."""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found at {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('NaN', 'null')
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        data = [json.loads(line) for line in content.splitlines() if line.strip()]
        return data

def read_embedded_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Reads and parses embedded JSON files that contain small_embedding field."""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: Embedded file not found at {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('NaN', 'null')
    
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        data = [json.loads(line) for line in content.splitlines() if line.strip()]
        return data

def extract_meaningful_text(record: Dict[str, Any]) -> str:
    """
    Intelligently extracts readable text from a record, parsing nested JSON if necessary.
    Handles complex Reddit/YouTube data structures.
    """
    # Try multiple text sources
    text_sources = [
        record.get('selftext'),
        record.get('text'),
        record.get('raw_text'),
        record.get('title'),
        record.get('body'),
        record.get('content'),
        record.get('data', {}).get('title'),
        record.get('data', {}).get('selftext'),
    ]
    
    for text_content in text_sources:
        if text_content and isinstance(text_content, str) and text_content.strip():
            # If it looks like JSON, try to parse it
            if text_content.strip().startswith('{'):
                try:
                    nested_data = json.loads(text_content)
                    title = nested_data.get('title', '')
                    description = nested_data.get('description', '')
                    body = nested_data.get('body', '')
                    
                    full_text = f"{title} {description} {body}".strip()
                    if full_text:
                        return full_text[:500]
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Return the text as-is (truncated)
            return str(text_content)[:500]
    
    # Fallback: extract summary from record structure
    fallback_fields = ['subreddit', 'author', 'username', 'channel', 'platform']
    fallback_parts = []
    for field in fallback_fields:
        if record.get(field):
            fallback_parts.append(str(record.get(field)))
    
    return " ".join(fallback_parts) if fallback_parts else "No content available"

def find_best_cluster_match(cluster_name: str, available_clusters: List[str], threshold: int = 80) -> str:
    """
    Finds the best matching cluster name using fuzzy string matching.
    Returns the best match if similarity is above threshold, otherwise returns None.
    """
    if not cluster_name or not available_clusters:
        return None
    
    best_match, score = process.extractOne(cluster_name, available_clusters, scorer=fuzz.partial_ratio)
    
    if score >= threshold:
        print(f"✅ Fuzzy matched '{cluster_name}' to '{best_match}' with score {score}")
        return best_match
    else:
        print(f"❌ No good match found for '{cluster_name}'. Best was '{best_match}' with score {score}")
        return None

def format_data_for_llm(record: Dict[str, Any]) -> str:
    """
    Formats a record with only the most important fields for the LLM context.
    Uses username and title instead of generic "Item X" format.
    """
    formatted_text = ""
    
    # Add username if available
    username = record.get('username') or record.get('author') or record.get('user') or 'Unknown User'
    formatted_text += f"User: {username}\n"
    
    # Add title if available
    title = record.get('title') or record.get('post_title') or ''
    if title:
        if len(title) > MAX_TEXT_LENGTH:
            title = title[:MAX_TEXT_LENGTH] + "..."
        formatted_text += f"Title: {title}\n"
    
    # Add source if available
    if 'source' in record and record['source']:
        formatted_text += f"Source: {record['source']}\n"
    
    # Add engagement metrics
    engagement_metrics = []
    engagement_fields = ['score', 'upvotes', 'downvotes', 'likes', 'comments', 'views', 'engagement', 'retweets', 'shares']
    
    for metric in engagement_fields:
        if metric in record and record[metric] is not None:
            engagement_metrics.append(f"{metric}: {record[metric]}")
    
    if engagement_metrics:
        formatted_text += f"Engagement: {', '.join(engagement_metrics)}\n"
    
    # Add content if available (truncated)
    text_content = extract_meaningful_text(record)
    if text_content and not title:  # Only add content if we don't have a title
        if len(text_content) > MAX_TEXT_LENGTH:
            text_content = text_content[:MAX_TEXT_LENGTH] + "..."
        formatted_text += f"Content: {text_content}\n"
    
    return formatted_text.strip()

async def get_cluster_name_from_llm(texts: List[str], cluster_id: int) -> str:
    """Asks the Llama model to generate a concise name for a cluster based on sample texts."""
    # Filter out empty or very short texts
    valid_texts = [text for text in texts if text and len(str(text).strip()) > 20]
    
    if not valid_texts:
        return f"Cluster {cluster_id}"

    # Use first 10 texts as samples for better naming
    sample_texts = "\n".join(f"- {text[:200].strip()}" for text in valid_texts[:10])
    
    prompt = f"""You are an expert data analyst. Based on the following sample texts from a single data cluster, provide a SHORT, DESCRIPTIVE name for the main topic or theme.

The name must be 2-5 words maximum and must describe the core subject matter or theme of the texts.

Sample Texts:
{sample_texts}

Your response MUST be ONLY a valid JSON object with this exact format:
{{"cluster_name": "Your Topic Name Here"}}

Examples of good cluster names:
- "Political Discussions & Activism"
- "Technology & Innovation News"
- "Mental Health Support Communities"
- "Gaming & Esports"
- "Environmental Justice"

Do NOT respond with anything other than the JSON object.
"""
    try:
        client = get_groq_client()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=50,
        )
        response_content = chat_completion.choices[0].message.content.strip()
        result = json.loads(response_content)
        cluster_name = result.get("cluster_name", f"Cluster {cluster_id}")
        
        # Ensure it's not just "Cluster N"
        if cluster_name.startswith("Cluster ") and cluster_name[8:].isdigit():
            print(f"⚠️ Warning: LLM returned generic name, retrying with different samples...")
            return f"Topic {cluster_id}"
        
        return cluster_name
    except Exception as e:
        print(f"Error calling LLM for cluster {cluster_id}: {e}")
        return f"Topic {cluster_id}"

async def chat_with_clustered_data(question: str, cluster_data: List[Dict[str, Any]], cluster_name: str = None) -> str:
    """Uses the Llama model to answer questions about the clustered data."""
    cluster_summary = {}
    
    if cluster_name:
        filtered_data = [item for item in cluster_data if item.get('cluster_name') == cluster_name]
        if not filtered_data:
            return f"I couldn't find any data for the cluster '{cluster_name}'. Available clusters are: {', '.join(set([item.get('cluster_name', 'Unknown') for item in cluster_data]))}"
        
        cluster_data = filtered_data
        context_header = f"Data from the '{cluster_name}' cluster:\n\n"
    else:
        context_header = "All clustered data:\n\n"
    
    for item in cluster_data:
        cluster_id = item.get('cluster_id', -1)
        item_cluster_name = item.get('cluster_name', f'Cluster {cluster_id}')
        
        if item_cluster_name not in cluster_summary:
            cluster_summary[item_cluster_name] = []
        
        formatted_item = format_data_for_llm(item)
        if formatted_item:
            cluster_summary[item_cluster_name].append(formatted_item)
    
    context = context_header
    
    for cluster_name, items in cluster_summary.items():
        context += f"Cluster: {cluster_name}\n"
        context += "Items:\n"
        for i, item_data in enumerate(items[:MAX_ITEMS_PER_CLUSTER]):
            # Extract username from the formatted data for better identification
            username_match = re.search(r'User: (.+)', item_data)
            username = username_match.group(1) if username_match else f"Item {i+1}"
            
            context += f"{username}:\n{item_data}\n\n"
        context += "\n"
    
    model_to_use = "llama-3.1-8b-instant"
    
    prompt = f"""
    You are a data analyst assistant. Based on the following clustered data, answer the user's question.
    Each item includes the username, title/content, and engagement metrics (likes, comments, views, etc.) when available.
    
    {context}
    
    Question: {question}
    
    Please provide a comprehensive answer based on the data. Analyze engagement metrics when relevant to the question.
    When referring to specific items, use the username and title information instead of generic "Item X" format.
    If the question cannot be answered with the available data, politely explain that and suggest what kind of data would be needed to answer it.
    
    Your response should be in a conversational tone and include insights about the clusters when relevant.
    Keep your response concise and focused on the question.
    """
    
    try:
        client = get_groq_client()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_to_use,
            temperature=0.3,
            max_tokens=500,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM for chat: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again later."

# --- API Endpoints ---
@app.post("/cluster")
async def cluster_data(num_clusters: int = 8):
    """
    Loads data from data.jsonl, generates embeddings, clusters it, and uses an LLM to generate cluster names.
    """
    print("🚀 Starting clustering process with data.jsonl...")
    all_records = []
    embeddings_list = []
    
    # Load data from data.jsonl with streaming to handle large files
    if not DATA_FILE_PATH.exists():
        raise HTTPException(status_code=404, detail=f"data.jsonl not found at {DATA_FILE_PATH}")
    
    print(f"📂 Loading data from {DATA_FILE_PATH}...")
    record_count = 0
    
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if record_count >= MAX_RECORDS_PER_CLUSTERING:
                    print(f"  ⏳ Reached maximum records limit ({MAX_RECORDS_PER_CLUSTERING})")
                    break
                
                if line.strip():
                    try:
                        record = json.loads(line)
                        all_records.append(record)
                        record_count += 1
                        
                        # Extract text for embedding
                        text_to_embed = (record.get("text") or record.get("content") or 
                                       record.get("body") or record.get("raw_text") or 
                                       record.get("title") or "")
                        
                        if not text_to_embed:
                            # Fallback: use JSON string representation
                            text_to_embed = json.dumps(record)[:500]
                        
                        # Generate embedding
                        embedding_model = get_embedding_model()
                        embedding = embedding_model.encode(str(text_to_embed))
                        embeddings_list.append(embedding)
                        
                        if record_count % 500 == 0:
                            print(f"  ⏳ Processed {record_count} records...")
                            
                    except json.JSONDecodeError as e:
                        if line_num <= 5:
                            print(f"  ⚠️ Skipping malformed JSON at line {line_num}")
                            
    except Exception as e:
        print(f"❌ ERROR reading data file: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading data file: {e}")
    
    if not all_records or not embeddings_list:
        raise HTTPException(status_code=404, detail="No data loaded from data.jsonl")

    print(f"📊 Total records loaded: {len(all_records)}")
    print(f"🧠 Generating embeddings for {len(embeddings_list)} records...")
    
    # Convert embeddings to numpy array
    embeddings = np.array(embeddings_list)
    print(f"🧠 Using embeddings with shape: {embeddings.shape}")

    print(f"🔄 Performing KMeans clustering with {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    cluster_assignments = kmeans.fit_predict(embeddings)

    clustered_data = {i: [] for i in range(num_clusters)}
    for i, record in enumerate(all_records):
        cluster_id = int(cluster_assignments[i])
        record['cluster_id'] = cluster_id
        clustered_data[cluster_id].append(record)

    print("🤖 Asking Llama to name the clusters...")
    cluster_name_map = {}
    for cluster_id, items in clustered_data.items():
        sample_texts = [extract_meaningful_text(item) for item in items]
        cluster_name = await get_cluster_name_from_llm(sample_texts, cluster_id)
        cluster_name_map[cluster_id] = cluster_name
        print(f"  - Cluster {cluster_id} named: '{cluster_name}'")
        await asyncio.sleep(0.5)

    print("✅ Cluster names generated:", cluster_name_map)

    final_clustered_records = []
    cluster_summary = []
    for cluster_id, name in cluster_name_map.items():
        items = clustered_data.get(cluster_id, [])
        for item in items:
            item['cluster_name'] = name
            final_clustered_records.append(item)
        cluster_summary.append({"cluster_id": cluster_id, "cluster_name": name, "item_count": len(items)})

    # Cache the clustered data for chat functionality
    global clustered_data_cache, cluster_names_cache
    clustered_data_cache = final_clustered_records
    cluster_names_cache = list(set([item['cluster_name'] for item in final_clustered_records]))
    
    print(f"📊 Available clusters: {cluster_names_cache}")

    return {"clusters": cluster_summary, "data": final_clustered_records}

@app.post("/chat")
async def chat_with_data(chat_request: ChatRequest):
    """
    Allows users to ask questions about the clustered data, optionally focusing on a specific cluster.
    """
    if not chat_request.cluster_data and not clustered_data_cache:
        raise HTTPException(
            status_code=400, 
            detail="No clustered data available. Please run the /cluster endpoint first or provide data in the request."
        )
    
    data_to_use = chat_request.cluster_data if chat_request.cluster_data else clustered_data_cache
    
    if not data_to_use:
        raise HTTPException(
            status_code=400, 
            detail="No clustered data available. Please run the /cluster endpoint first."
        )
    
    target_cluster_name = None
    
    if chat_request.cluster_name:
        available_clusters = list(set([item.get('cluster_name') for item in data_to_use]))
        target_cluster_name = find_best_cluster_match(chat_request.cluster_name, available_clusters)
        
        if not target_cluster_name:
            available_clusters_str = ", ".join(available_clusters)
            return {
                "question": chat_request.question, 
                "cluster_name": chat_request.cluster_name,
                "response": f"I couldn't find a cluster named '{chat_request.cluster_name}'. Available clusters are: {available_clusters_str}"
            }
    
    print(f"💬 Processing chat question: '{chat_request.question}' for cluster: '{target_cluster_name if target_cluster_name else 'All clusters'}'")
    
    try:
        response = await chat_with_clustered_data(
            chat_request.question, 
            data_to_use, 
            target_cluster_name
        )
        return {
            "question": chat_request.question,
            "cluster_name": target_cluster_name,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/clusters")
async def get_available_clusters():
    """Returns the list of available cluster names from the cached data."""
    if not clustered_data_cache:
        raise HTTPException(
            status_code=400, 
            detail="No clustered data available. Please run the /cluster endpoint first."
        )
    
    cluster_names = list(set([item.get('cluster_name') for item in clustered_data_cache]))
    return {"clusters": cluster_names}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Clustering API. Use the /cluster endpoint to process data and /chat to ask questions about it."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)