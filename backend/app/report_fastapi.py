import json
import os
import re
import numpy as np
from fastapi import FastAPI, HTTPException
from neo4j import GraphDatabase, exceptions
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import uuid
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from portable path
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Read credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Global Variables for AI Agent ---
data_store = []
embedding_store = None
knn_model = None
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Lazy loading function for Groq client
def get_groq_client():
    """Initialize and return Groq client with error handling"""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")
    return Groq(api_key=GROQ_API_KEY)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="SocialPulse Simppl Task API",
    description="An API to populate a Neo4j database and chat with your data.",
    version="3.0.1"  # Updated version for network graph features
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatQuery(BaseModel):
    query: str

# --- Helper Functions ---
def read_json_file(file_path):
    """Read JSON file with streaming support for large files to avoid MemoryError"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to read as a single JSON array first (for smaller files)
            first_char = f.read(1)
            if not first_char:
                return None
            f.seek(0)
            
            if first_char == '[':
                # It's a JSON array - try to load it
                try:
                    file_size = f.seek(0, 2)  # Seek to end to get size
                    f.seek(0)
                    
                    # If file is larger than 100MB, use streaming
                    if file_size > 100 * 1024 * 1024:
                        print(f"Large file detected ({file_size / 1024 / 1024:.1f}MB), using streaming parser...")
                        return read_json_array_streaming(file_path)
                    
                    content = f.read()
                    content = content.replace('NaN', 'null')
                    data = json.loads(content)
                    return data if isinstance(data, list) else [data]
                except json.JSONDecodeError:
                    print(f"Failed to parse as JSON array, trying JSON Lines format...")
                    return read_json_lines(file_path)
            else:
                # Likely JSON Lines format
                return read_json_lines(file_path)
                
    except FileNotFoundError:
        print(f"ERROR: File not found at path: {file_path}")
        return None
    except MemoryError:
        print(f"MemoryError detected for {Path(file_path).name}, switching to streaming parser...")
        return read_json_array_streaming(file_path)

def read_json_array_streaming(file_path):
    """Read large JSON arrays in chunks to avoid MemoryError"""
    data = []
    chunk_size = 65536  # Read 64KB at a time (larger chunks = faster)
    buffer = ""
    depth = 0
    in_string = False
    escape_next = False
    record_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                buffer += chunk
                i = 0
                while i < len(buffer):
                    char = buffer[i]
                    
                    # Handle string literals
                    if char == '"' and not escape_next:
                        in_string = not in_string
                    elif char == '\\' and in_string:
                        escape_next = not escape_next
                    elif not in_string and char == escape_next:
                        escape_next = False
                    elif not in_string:
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                # Found a complete object
                                json_str = buffer[0:i+1].strip()
                                if json_str.startswith(','):
                                    json_str = json_str[1:].strip()
                                if json_str:
                                    try:
                                        json_str = json_str.replace('NaN', 'null')
                                        obj = json.loads(json_str)
                                        data.append(obj)
                                        record_count += 1
                                        if record_count % 1000 == 0:
                                            print(f"  ⏳ Streaming: Loaded {record_count} records...")
                                    except json.JSONDecodeError:
                                        pass
                                buffer = buffer[i+1:].lstrip(',').lstrip()
                                i = -1
                    
                    i += 1
    except Exception as e:
        print(f"Error reading JSON stream: {e}")
    
    print(f"  ✅ Streaming complete: Loaded {len(data)} total records.")
    return data if data else None

def read_json_lines(file_path):
    """Read JSON Lines format (one JSON object per line)"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        line = line.replace('NaN', 'null')
                        obj = json.loads(line)
                        data.append(obj)
                    except json.JSONDecodeError as e:
                        if line_num <= 5:  # Only print first 5 errors
                            print(f"Skipping malformed line {line_num}: {line[:100]}... Error: {e}")
        return data if data else None
    except Exception as e:
        print(f"Error reading JSON Lines: {e}")
        return None

def parse_embedding(embedding_str: str):
    try:
        if isinstance(embedding_str, str) and embedding_str.startswith('[') and embedding_str.endswith(']'):
            return json.loads(embedding_str)
        elif isinstance(embedding_str, list):
            return embedding_str
    except (json.JSONDecodeError, TypeError):
        return None
    return None

def validate_and_clean_visualizations(llm_output: dict):
    """
    Enhanced validation for visualizations from the LLM output.
    Ensures proper structure and data integrity for frontend plotting.
    """
    if "visualizations" not in llm_output or not isinstance(llm_output.get("visualizations"), list):
        llm_output["visualizations"] = []
        return llm_output

    valid_charts = []
    for chart_idx, chart in enumerate(llm_output["visualizations"]):
        if not isinstance(chart, dict):
            print(f"⚠️ Warning: Discarding non-dict visualization: {chart}")
            continue
            
        chart_type = chart.get("type", "").lower()
        
        # Validate required fields based on chart type
        if chart_type in ["bar", "line", "pie"]:
            labels = chart.get("labels", [])
            data = chart.get("data", [])
            
            if not isinstance(labels, list):
                print(f"⚠️ Warning: Labels is not a list: {labels}")
                continue
                
            # Clean and validate data array
            if isinstance(data, str):
                try:
                    if re.match(r'^[\d.]+$', data):
                        parts = re.findall(r'\d+\.\d+|\d+', data)
                        data = [float(part) for part in parts if part and float(part) > 0]
                    else:
                        data = json.loads(data)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"⚠️ Warning: Could not parse data string '{data}': {e}")
                    continue
            
            if not isinstance(data, list):
                print(f"⚠️ Warning: Data is not a list: {data}")
                continue
                
            # Clean data values
            cleaned_data = []
            for value in data:
                try:
                    if value is None or value == "":
                        cleaned_data.append(0)
                    elif isinstance(value, str):
                        num_val = float(value)
                        cleaned_data.append(num_val)
                    else:
                        cleaned_data.append(float(value))
                except (ValueError, TypeError):
                    print(f"⚠️ Warning: Invalid data value '{value}', replacing with 0")
                    cleaned_data.append(0)
            
            data = cleaned_data
            
            # Ensure labels and data have the same length
            if len(labels) != len(data):
                print(f"⚠️ Warning: Chart {chart_idx + 1} - Labels ({len(labels)}) and data ({len(data)}) length mismatch")
                min_length = min(len(labels), len(data))
                if min_length > 0:
                    labels = labels[:min_length]
                    data = data[:min_length]
                    print(f"✅ Fixed by truncating to length {min_length}")
                else:
                    print("❌ Cannot fix - skipping chart")
                    continue
            
            if len(labels) == 0 or len(data) == 0:
                print(f"⚠️ Warning: Empty chart data - skipping")
                continue
                
            # Update chart with cleaned data
            chart["labels"] = labels
            chart["data"] = data
            
            # Add default colors if not present
            if "colors" not in chart or len(chart.get("colors", [])) != len(labels):
                default_colors = [
                    "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
                    "#F7DC6F", "#BB8FCE", "#85C1E9", "#F8C471", "#82E0AA",
                    "#FFD93D", "#6BCF7F", "#4D96FF", "#9B59B6", "#E67E22",
                    "#1ABC9C", "#E74C3C", "#3498DB", "#F39C12", "#27AE60"
                ]
                chart["colors"] = (default_colors * ((len(labels) // len(default_colors)) + 1))[:len(labels)]
            
            # Ensure required metadata exists
            if "title" not in chart or not chart["title"]:
                chart["title"] = f"{chart_type.title()} Chart"
            if "description" not in chart or not chart["description"]:
                chart["description"] = f"Data visualization showing {chart_type} chart"
            if "id" not in chart or not chart["id"]:
                chart["id"] = f"chart_{len(valid_charts) + 1}"
                
            # Add axis labels if missing
            if "xAxisLabel" not in chart:
                chart["xAxisLabel"] = "Categories"
            if "yAxisLabel" not in chart:
                chart["yAxisLabel"] = "Values"
                
            print(f"✅ Valid chart added: {chart['title']} ({len(labels)} data points)")
            valid_charts.append(chart)
            
        else:
            print(f"⚠️ Warning: Unsupported chart type '{chart_type}': {chart}")
    
    llm_output["visualizations"] = valid_charts
    print(f"📊 Final result: {len(valid_charts)} valid visualizations out of {len(llm_output.get('visualizations', []))} attempted")
    return llm_output

def extract_topics_from_text(text: str) -> List[str]:
    """Extract potential topics from text using simple keyword matching"""
    if not text:
        return []
    
    text_lower = text.lower()
    topics = []
    
    # Define topic keywords
    topic_keywords = {
        "technology": ["ai", "artificial intelligence", "machine learning", "tech", "computer", "software", "hardware", "algorithm", "data science"],
        "gaming": ["game", "gaming", "playstation", "xbox", "nintendo", "steam", "gamer", "esports"],
        "entertainment": ["movie", "film", "tv", "show", "celebrity", "hollywood", "netflix", "youtube", "streaming"],
        "science": ["science", "scientific", "research", "physics", "chemistry", "biology", "space", "nasa"],
        "politics": ["politics", "government", "election", "democrat", "republican", "policy", "law"],
        "sports": ["sports", "basketball", "football", "soccer", "baseball", "tennis", "olympics"],
        "health": ["health", "medical", "medicine", "doctor", "hospital", "fitness", "diet", "nutrition"],
        "education": ["education", "school", "university", "college", "student", "teacher", "learning"],
        "business": ["business", "economy", "market", "finance", "investment", "stock", "company"],
        "lifestyle": ["lifestyle", "travel", "food", "cooking", "fashion", "beauty", "home"]
    }
    
    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                topics.append(topic)
                break  # Only add topic once per text
    
    return topics[:3]  # Limit to 3 most relevant topics

# --- Neo4j Driver and Data Loading ---
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def create_constraints(tx):
    """Create constraints for all node types"""
    constraints = [
        "CREATE CONSTRAINT user_uniqueness IF NOT EXISTS FOR (u:User) REQUIRE u.username IS UNIQUE",
        "CREATE CONSTRAINT reddit_post_uniqueness IF NOT EXISTS FOR (p:RedditPost) REQUIRE p.internal_id IS UNIQUE",
        "CREATE CONSTRAINT youtube_post_uniqueness IF NOT EXISTS FOR (p:YouTubePost) REQUIRE p.internal_id IS UNIQUE",
        "CREATE CONSTRAINT reddit_comment_uniqueness IF NOT EXISTS FOR (c:RedditComment) REQUIRE c.comment_id IS UNIQUE",
        "CREATE CONSTRAINT youtube_comment_uniqueness IF NOT EXISTS FOR (c:YouTubeComment) REQUIRE c.comment_id IS UNIQUE",
        "CREATE CONSTRAINT cluster_uniqueness IF NOT EXISTS FOR (c:Cluster) REQUIRE c.cluster_id IS UNIQUE",
        "CREATE CONSTRAINT topic_uniqueness IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE"
    ]
    
    for constraint in constraints:
        try:
            tx.run(constraint)
        except Exception as e:
            print(f"Warning: Could not create constraint: {e}")

def create_clusters_and_topics(tx):
    """Create cluster and topic nodes with hierarchical relationships"""
    # Create platform clusters
    platform_clusters = {
        "YouTube": "youtube_platform",
        "Reddit": "reddit_platform",
        "YouTube_Posts": "youtube_posts",
        "YouTube_Comments": "youtube_comments", 
        "Reddit_Posts": "reddit_posts",
        "Reddit_Comments": "reddit_comments"
    }
    
    for cluster_name, cluster_id in platform_clusters.items():
        tx.run("""
            MERGE (c:Cluster {cluster_id: $cluster_id})
            SET c.name = $cluster_name, c.type = 'platform', c.created_at = datetime()
        """, cluster_id=cluster_id, cluster_name=cluster_name)
    
    # Create hierarchical relationships
    hierarchy = [
        ("youtube_posts", "youtube_platform", "PART_OF"),
        ("youtube_comments", "youtube_platform", "PART_OF"),
        ("reddit_posts", "reddit_platform", "PART_OF"),
        ("reddit_comments", "reddit_platform", "PART_OF")
    ]
    
    for child_id, parent_id, rel_type in hierarchy:
        tx.run("""
            MATCH (child:Cluster {cluster_id: $child_id})
            MATCH (parent:Cluster {cluster_id: $parent_id})
            MERGE (child)-[r:PART_OF]->(parent)
            SET r.created_at = datetime()
        """, child_id=child_id, parent_id=parent_id)
    
    # Create common topics
    common_topics = [
        "technology", "gaming", "entertainment", "science", "politics",
        "sports", "health", "education", "business", "lifestyle"
    ]
    
    for topic in common_topics:
        tx.run("""
            MERGE (t:Topic {name: $topic})
            SET t.created_at = datetime()
        """, topic=topic)

def load_data_with_clusters_and_topics(tx, records, platform):
    """Load data with cluster assignments and topic relationships"""
    if not isinstance(records, list): 
        return
    
    for record in records:
        # Extract basic data
        timestamp = record.get('timestamp') or record.get('date_of_comment')
        formatted_timestamp = timestamp.replace(' ', 'T', 1) if timestamp else None
        reactions = json.loads(record.get('reactions', '{}')) if isinstance(record.get('reactions'), str) else record.get('reactions', {})
        text_analysis = json.loads(record.get('text_analysis', '{}')) if isinstance(record.get('text_analysis'), str) else record.get('text_analysis', {})
        
        # Extract text for topic analysis
        text_content = record.get('text', '') or record.get('raw_text', '') or record.get('title', '')
        topics = extract_topics_from_text(text_content)
        
        # Determine cluster based on platform and type
        if platform == "RedditPost":
            cluster_id = "reddit_posts"
        elif platform == "YouTubePost":
            cluster_id = "youtube_posts"
        elif platform == "RedditComment":
            cluster_id = "reddit_comments"
        elif platform == "YouTubeComment":
            cluster_id = "youtube_comments"
        else:
            cluster_id = "unknown"
        
        params = {
            "internal_id": record.get("id"),
            "username": record.get("username"),
            "timestamp": formatted_timestamp,
            "topics": topics,
            "cluster_id": cluster_id,
            "text_content": text_content[:500]  # Limit text length
        }
        
        # Add platform-specific parameters
        if platform in ["RedditPost", "YouTubePost"]:
            params.update({
                "post_id": record.get("post_id"),
                "link": record.get("link")
            })
        elif platform in ["RedditComment", "YouTubeComment"]:
            params.update({
                "comment_id": record.get("comment_id"),
                "post_id": record.get("post_id")
            })
        
        # Build node properties
        node_props = {
            "likes": reactions.get("likes"),
            "engagement": record.get("engagement"),
            "text": text_content[:1000],  # Limit text length
            "topics": topics
        }
        
        # Add sentiment and emotion data
        sentiment_data = text_analysis.get("Sentiment", {})
        if sentiment_data:
            for key, value in sentiment_data.items(): 
                node_props[f"sentiment_{key}"] = value
        
        emotion_data = text_analysis.get("Emotion", {})
        if emotion_data:
            for key, value in emotion_data.items(): 
                node_props[f"emotion_{key}"] = value
        
        if platform == "YouTubePost":
            node_props["views"] = record.get("views")
        
        params["node_props"] = node_props
        
        # Create the main node - SEPARATE QUERIES to avoid syntax issues
        if platform == "RedditPost":
            # First create the post and user
            tx.run("""
                MERGE (u:User {username: $username})
                MERGE (p:RedditPost {internal_id: $internal_id})
                SET p.reddit_id = $post_id, p.link = $link, p.timestamp = datetime($timestamp)
                SET p += $node_props
                MERGE (u)-[:POSTED]->(p)
            """, **params)
            
            # Then connect to cluster
            tx.run("""
                MATCH (p:RedditPost {internal_id: $internal_id})
                MATCH (c:Cluster {cluster_id: $cluster_id})
                MERGE (p)-[:BELONGS_TO]->(c)
            """, **params)
            
            # Then connect to topics
            for topic in topics:
                tx.run("""
                    MATCH (p:RedditPost {internal_id: $internal_id})
                    MERGE (t:Topic {name: $topic})
                    MERGE (p)-[:ABOUT]->(t)
                """, internal_id=params["internal_id"], topic=topic)
                
        elif platform == "YouTubePost":
            # First create the post and user
            tx.run("""
                MERGE (u:User {username: $username})
                MERGE (p:YouTubePost {internal_id: $internal_id})
                SET p.video_id = $post_id, p.link = $link, p.timestamp = datetime($timestamp)
                SET p += $node_props
                MERGE (u)-[:POSTED]->(p)
            """, **params)
            
            # Then connect to cluster
            tx.run("""
                MATCH (p:YouTubePost {internal_id: $internal_id})
                MATCH (c:Cluster {cluster_id: $cluster_id})
                MERGE (p)-[:BELONGS_TO]->(c)
            """, **params)
            
            # Then connect to topics
            for topic in topics:
                tx.run("""
                    MATCH (p:YouTubePost {internal_id: $internal_id})
                    MERGE (t:Topic {name: $topic})
                    MERGE (p)-[:ABOUT]->(t)
                """, internal_id=params["internal_id"], topic=topic)
                
        elif platform == "RedditComment":
            # First create the comment and relationships
            tx.run("""
                MATCH (post:RedditPost {internal_id: $post_id})
                MERGE (u:User {username: $username})
                MERGE (c:RedditComment {comment_id: $comment_id})
                SET c.date = datetime($timestamp)
                SET c += $node_props
                MERGE (u)-[:WROTE]->(c)
                MERGE (c)-[:REPLY_TO]->(post)
            """, **params)
            
            # Then connect to cluster
            tx.run("""
                MATCH (c:RedditComment {comment_id: $comment_id})
                MATCH (cluster:Cluster {cluster_id: $cluster_id})
                MERGE (c)-[:BELONGS_TO]->(cluster)
            """, **params)
            
            # Then connect to topics
            for topic in topics:
                tx.run("""
                    MATCH (c:RedditComment {comment_id: $comment_id})
                    MERGE (t:Topic {name: $topic})
                    MERGE (c)-[:ABOUT]->(t)
                """, comment_id=params["comment_id"], topic=topic)
                
        elif platform == "YouTubeComment":
            # First create the comment and relationships
            tx.run("""
                MATCH (post:YouTubePost {internal_id: $post_id})
                MERGE (u:User {username: $username})
                MERGE (c:YouTubeComment {comment_id: $comment_id})
                SET c.date = datetime($timestamp)
                SET c += $node_props
                MERGE (u)-[:WROTE]->(c)
                MERGE (c)-[:REPLY_TO]->(post)
            """, **params)
            
            # Then connect to cluster
            tx.run("""
                MATCH (c:YouTubeComment {comment_id: $comment_id})
                MATCH (cluster:Cluster {cluster_id: $cluster_id})
                MERGE (c)-[:BELONGS_TO]->(cluster)
            """, **params)
            
            # Then connect to topics
            for topic in topics:
                tx.run("""
                    MATCH (c:YouTubeComment {comment_id: $comment_id})
                    MERGE (t:Topic {name: $topic})
                    MERGE (c)-[:ABOUT]->(t)
                """, comment_id=params["comment_id"], topic=topic)

def query_neo4j_for_context(query_text: str, limit: int = 10) -> List[Dict]:
    """Query Neo4j for relevant context based on the query"""
    try:
        with driver.session() as session:
            # Simple keyword-based search for now
            result = session.run("""
                MATCH (n)
                WHERE n.text CONTAINS $search_query OR 
                      ANY(topic IN n.topics WHERE topic CONTAINS $search_query) OR
                      EXISTS(n.sentiment_compound) OR
                      EXISTS(n.engagement)
                RETURN n
                LIMIT $result_limit
            """, search_query=query_text.lower(), result_limit=limit)
            
            context_records = []
            for record in result:
                node = record["n"]
                node_data = dict(node)
                
                # Extract relevant information based on node type
                context_record = {
                    "platform": "Reddit" if "Reddit" in node.labels else "YouTube" if "YouTube" in node.labels else "Unknown",
                    "type": "Post" if "Post" in node.labels else "Comment" if "Comment" in node.labels else "Unknown",
                    "username": node_data.get("username"),
                    "text": node_data.get("text", "")[:500],  # Limit text length
                    "engagement": node_data.get("engagement"),
                    "likes": node_data.get("likes"),
                    "timestamp": str(node_data.get("timestamp")) if node_data.get("timestamp") else None,
                    "topics": node_data.get("topics", [])
                }
                
                # Add sentiment data if available
                sentiment_keys = [k for k in node_data.keys() if k.startswith("sentiment_")]
                if sentiment_keys:
                    context_record["sentiment"] = {k: node_data[k] for k in sentiment_keys}
                
                context_records.append(context_record)
            
            return context_records
            
    except Exception as e:
        print(f"Error querying Neo4j: {e}")
        return []

# --- Application Startup Event ---
@app.on_event("startup")
def startup_event():
    global data_store, embedding_store, knn_model
    print("🚀 Application startup: Loading data and generating embeddings...")
    
    # Path to data.jsonl file
    DATA_FILE_PATH = Path(__file__).parent.parent / "data" / "data.jsonl"
    
    print(f"🔄 Loading data from {DATA_FILE_PATH.name}...")
    temp_embeddings = []
    MAX_RECORDS = 2000
    record_count = 0
    empty_text_count = 0
    
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if record_count >= MAX_RECORDS:
                    print(f"  ⏳ Reached maximum records limit ({MAX_RECORDS})")
                    break
                
                if line.strip():
                    try:
                        record = json.loads(line)
                        data_store.append(record)
                        record_count += 1
                        
                        # Generate embedding from text content - use empty string if no text
                        text_to_embed = (record.get("text") or record.get("content") or 
                                       record.get("body") or record.get("raw_text") or 
                                       record.get("title") or "")
                        
                        if not text_to_embed:
                            empty_text_count += 1
                            # Generate embedding from any available fields as fallback
                            text_to_embed = json.dumps(record)[:500] if record else ""
                        
                        if text_to_embed:
                            embedding = embedding_model.encode(str(text_to_embed))
                            temp_embeddings.append(embedding)
                        
                        if record_count % 500 == 0:
                            print(f"  ⏳ Processed {record_count} records...")
                            
                    except json.JSONDecodeError as e:
                        if line_num <= 5:
                            print(f"  ⚠️ Skipping malformed JSON at line {line_num}")
                            
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {DATA_FILE_PATH}")
        print(f"❌ CRITICAL ERROR: Cannot load data. Chat agent will not work.")
        return
    except Exception as e:
        print(f"❌ ERROR reading data file: {e}")
        return
    
    print(f"  📊 Statistics: {record_count} records loaded, {empty_text_count} with empty text fields")
    
    if not temp_embeddings or not data_store:
        print("❌ CRITICAL ERROR: No records loaded. Chat agent will not work.")
        return
    
    embedding_store = np.array(temp_embeddings)
    knn_model = NearestNeighbors(n_neighbors=min(5, len(data_store)), metric='cosine')
    knn_model.fit(embedding_store)
    print(f"✅ Startup complete. Loaded {len(data_store)} records and generated {len(temp_embeddings)} embeddings. Retrieval model is ready.")

# --- API Endpoints ---
@app.post("/c-database", status_code=200)
def populate_database():
    base_path = Path(__file__).parent.parent / "json_data"
    files_to_process = {
        base_path / "posts_reddit.json": "RedditPost",
        base_path / "posts_youtube.json": "YouTubePost",
        base_path / "comments_reddit.json": "RedditComment",
        base_path / "comments_youtube.json": "YouTubeComment",
    }
    
    try:
        with driver.session() as session:
            # Create constraints and cluster structure
            session.write_transaction(create_constraints)
            session.write_transaction(create_clusters_and_topics)
            print("Constraints and clusters created successfully.")
            
            # Load data
            for file_path, platform in files_to_process.items():
                print(f"Processing subset of {Path(file_path).name}...")
                records = read_json_file(str(file_path))
                if not records:
                    print(f"Warning: No data loaded from {Path(file_path).name}.")
                    continue
                subset = records
                session.write_transaction(load_data_with_clusters_and_topics, subset, platform)
                print(f"Successfully processed subset from {Path(file_path).name} with cluster assignments.")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    
    return {"message": "Database population with clusters and topics completed successfully."}

@app.post("/clear-database", status_code=200)
def clear_database():
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        return {"message": "Database cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.get("/cluster-info")
def get_cluster_info():
    """Get information about clusters and their contents"""
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Cluster)
                OPTIONAL MATCH (c)<-[:BELONGS_TO]-(n)
                RETURN c.cluster_id AS cluster_id, 
                       c.name AS cluster_name, 
                       c.type AS cluster_type,
                       COUNT(n) AS node_count
                ORDER BY c.type, c.name
            """)
            
            clusters = []
            for record in result:
                clusters.append({
                    "cluster_id": record["cluster_id"],
                    "name": record["cluster_name"],
                    "type": record["cluster_type"],
                    "node_count": record["node_count"]
                })
            
            return {"clusters": clusters}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cluster info: {e}")

@app.get("/topic-info")
def get_topic_info():
    """Get information about topics and their distribution"""
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (t:Topic)<-[:ABOUT]-(n)
                RETURN t.name AS topic_name, 
                       COUNT(n) AS mention_count,
                       COLLECT(DISTINCT labels(n)[0]) AS node_types
                ORDER BY mention_count DESC
                LIMIT 20
            """)
            
            topics = []
            for record in result:
                topics.append({
                    "topic_name": record["topic_name"],
                    "mention_count": record["mention_count"],
                    "node_types": record["node_types"]
                })
            
            return {"topics": topics}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving topic info: {e}")

@app.post("/chat")
async def chat_with_data(query: ChatQuery):
    if not data_store or embedding_store is None or knn_model is None:
        raise HTTPException(status_code=503, detail="RAG model is not ready. Please wait and try again.")
    
    query_embedding = embedding_model.encode(query.query)
    distances, indices = knn_model.kneighbors([query_embedding])
    context_records = [data_store[i] for i in indices[0]]
    
    cleaned_context = []
    for record in context_records:
        cleaned_record = {
            "platform": record.get("platform"),
            "username": record.get("username"),
            "text": record.get("text") or record.get("content") or record.get("body") or record.get("raw_text") or record.get("title"),
            "reactions": record.get("reactions"),
            "engagement": record.get("engagement"),
            "views": record.get("views"),
            "timestamp": record.get("timestamp") or record.get("date") or record.get("date_of_comment"),
            "text_analysis": record.get("text_analysis"),
            "sentiment": record.get("sentiment"),
            "emotion": record.get("emotion")
        }
        cleaned_context.append({k: v for k, v in cleaned_record.items() if v is not None})
        
    context_str = json.dumps(cleaned_context, indent=2)

    # --- Enhanced Comprehensive Prompt ---
    prompt = f"""You are a senior data analyst and insights expert. Analyze the provided data and answer the user's question comprehensively.

User Question: "{query.query}"

Data Context (Top 5 most relevant records):
```json
{context_str}
```

Provide your response as a JSON object with ALL these required keys:
1. natural_response: {{
     summary: "Brief overview of findings",
     key_points: ["point 1", "point 2", "point 3"],
     actionable_insights: ["insight 1", "insight 2"]
   }}
2. statistical_analysis: {{
     inferential_statistics: "Key statistical insight",
     correlation_analysis: "How variables relate",
     trend_analysis: "Observed trends",
     methodology: "Statistical methods used",
     data_quality_assessment: "Data quality notes"
   }}
3. detailed_report: {{
     comprehensive_analysis: "Full detailed analysis",
     limitations_and_caveats: "Known limitations",
     platform_comparison: "Platform comparison if applicable",
     user_behavior_analysis: "User behavior patterns",
     content_performance: "Content performance metrics"
   }}
4. visualizations: [array of 1-2 chart objects with: id, type (bar/line/pie), title, description, labels (array), data (array)]
5. metadata: {{
     analysis_timestamp: ISO timestamp,
     data_points_analyzed: number,
     confidence_level: "High/Medium/Low"
   }}
6. top_results: {{
     top_users: [{{ username: "...", reactions: number, rank: number }}, ...],
     top_posts: [{{ title: JSON.stringify({{ title: "..." }}), rank: number }}, ...],
     unique_users_count: number
   }}

Requirements:
- ALL 6 keys MUST be present in response
- Ensure labels and data arrays have EXACTLY the same length
- Use realistic numbers, not concatenated strings
- Include 1-2 meaningful visualizations with proper data
- Return valid JSON only
"""

    try:
        groq = get_groq_client()
        chat_completion = groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=4000
        )
        response_content = chat_completion.choices[0].message.content
        
        # Parse and validate the JSON from the LLM
        try:
            llm_output = json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to parse LLM response as JSON: {e}")
        
        # Ensure ALL required structure keys exist with defaults
        if "natural_response" not in llm_output:
            llm_output["natural_response"] = {
                "summary": "Analysis completed based on available data",
                "key_points": ["Data analyzed successfully"],
                "actionable_insights": ["Further analysis recommended"]
            }
        
        if "statistical_analysis" not in llm_output:
            llm_output["statistical_analysis"] = {
                "inferential_statistics": "Analysis of patterns in the data",
                "correlation_analysis": "Variables analyzed for relationships",
                "trend_analysis": "Temporal trends identified",
                "methodology": "RAG-based analysis with embedding similarity search",
                "data_quality_assessment": "Data processed successfully"
            }
        
        if "detailed_report" not in llm_output:
            llm_output["detailed_report"] = {
                "comprehensive_analysis": "Comprehensive analysis of the provided data context.",
                "limitations_and_caveats": "Analysis based on available data subset",
                "platform_comparison": "Multi-platform data analyzed",
                "user_behavior_analysis": "User engagement patterns identified",
                "content_performance": "Content metrics evaluated"
            }
        
        if "metadata" not in llm_output:
            llm_output["metadata"] = {
                "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
                "data_points_analyzed": len(context_records),
                "confidence_level": "Medium"
            }
        
        if "top_results" not in llm_output:
            # Generate top_results from context_records
            users_dict = {}
            for record in context_records:
                username = record.get("username", "Unknown")
                if username not in users_dict:
                    users_dict[username] = {"reactions": 0, "count": 0}
                reactions_data = record.get("reactions", {})
                if isinstance(reactions_data, dict):
                    total_reactions = sum(reactions_data.values()) if reactions_data else 0
                else:
                    total_reactions = 0
                users_dict[username]["reactions"] += total_reactions
                users_dict[username]["count"] += 1
            
            top_users = [
                {"username": user, "reactions": data["reactions"], "rank": i+1}
                for i, (user, data) in enumerate(sorted(users_dict.items(), 
                                                       key=lambda x: x[1]["reactions"], 
                                                       reverse=True)[:5])
            ]
            
            top_posts = [
                {"title": json.dumps({"title": record.get("text", record.get("title", ""))[:50]}), "rank": i+1}
                for i, record in enumerate(context_records[:5])
            ]
            
            llm_output["top_results"] = {
                "top_users": top_users,
                "top_posts": top_posts,
                "unique_users_count": len(users_dict)
            }
        
        if "visualizations" not in llm_output:
            llm_output["visualizations"] = []

        # Validate and clean visualizations
        validated_output = validate_and_clean_visualizations(llm_output)
        
        return validated_output

    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Insights-Plus API. Go to /docs to see the available endpoints."}

@app.get("/health")
def health_check():
    try:
        driver.verify_connectivity()
        neo4j_connected = True
    except:
        neo4j_connected = False
        
    return {
        "status": "healthy",
        "data_loaded": len(data_store),
        "model_ready": knn_model is not None,
        "neo4j_connected": neo4j_connected,
        "version": "3.0.1"
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Insights-Plus API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)