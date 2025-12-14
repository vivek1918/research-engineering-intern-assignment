import os
import json
import pandas as pd
import numpy as np
from fastapi import APIRouter
from sentence_transformers import SentenceTransformer
import logging

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the paths to your directories
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_DIR = os.path.join(BASE_DIR, "json_data")
FINAL_DIR = os.path.join(BASE_DIR, "final")
EMBEDDED_JSON_DIR = os.path.join(BASE_DIR, "embedded_json_data")  # New directory for embedded data

# Create the output directories if they don't exist
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(EMBEDDED_JSON_DIR, exist_ok=True)  # Create the new embedded directory

# Initialize the embedding model (small and efficient)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def truncate_text(text: str, max_tokens: int = 128) -> str:
    """Truncate text to reduce token count while preserving meaning"""
    if not isinstance(text, str):
        return ""
    
    # Simple token approximation
    tokens = text.split()
    if len(tokens) > max_tokens:
        # Keep the beginning and end for better context preservation
        half_tokens = max_tokens // 2
        truncated = " ".join(tokens[:half_tokens] + tokens[-half_tokens:])
        return truncated
    return text

def generate_small_embedding(text: str, max_tokens: int = 128) -> list:
    """Generate compact embeddings with truncated text"""
    if not text or not isinstance(text, str):
        # Return a small zero vector for empty text
        return [0.0] * 384
    
    try:
        # Truncate text to reduce token count
        truncated_text = truncate_text(text, max_tokens)
        
        # Generate embedding
        embedding = embedding_model.encode(truncated_text, show_progress_bar=False)
        
        # Reduce precision to float32 (smaller than float64, more stable than float16)
        embedding = embedding.astype(np.float32)
        
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return [0.0] * 384

def process_data_with_embeddings(df: pd.DataFrame, text_columns_config: dict = None) -> pd.DataFrame:
    """
    Process DataFrame: drop existing embeddings and generate new optimized ones
    """
    # Make a copy to avoid modifying original
    processed_df = df.copy()
    
    # Drop existing embedding columns
    embedding_cols = [col for col in processed_df.columns 
                     if any(keyword in col.lower() for keyword in ['embedding', 'vec'])]
    if embedding_cols:
        processed_df = processed_df.drop(columns=embedding_cols)
        logger.info(f"Dropped embedding columns: {embedding_cols}")
    
    # Auto-detect text columns for embedding if not specified
    text_columns = []
    if text_columns_config:
        text_columns = text_columns_config
    else:
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object' and not processed_df[col].empty:
                sample_values = processed_df[col].dropna()
                if not sample_values.empty:
                    sample_value = sample_values.iloc[0]
                    if isinstance(sample_value, str) and len(sample_value) > 5:
                        text_columns.append(col)
    
    # Generate embeddings for all text columns (concatenate for better context)
    if text_columns:
        logger.info(f"Generating embeddings for columns: {text_columns}")
        
        def combine_text_fields(row):
            """Combine multiple text fields for better embedding context"""
            combined_text = ""
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    combined_text += f" {str(row[col])}"
            return combined_text.strip()
        
        # Create combined text for embedding
        processed_df['combined_text'] = processed_df.apply(combine_text_fields, axis=1)
        
        # Generate embeddings
        logger.info("Generating small embeddings...")
        processed_df['small_embedding'] = processed_df['combined_text'].apply(
            lambda x: generate_small_embedding(x, max_tokens=150)
        )
        
        # Remove temporary combined text column
        processed_df = processed_df.drop(columns=['combined_text'])
    
    return processed_df

@router.get("/convert-and-save-all-data")
async def convert_and_save_all_data():
    """
    Converts all CSV files to JSON with new small embeddings and saves them
    """
    files_config = {
        "posts_reddit.json": {
            "csv_file": "posts Data Dump - Reddit.csv",
            "text_columns": ["title", "body", "content"]  # Specify likely text columns
        },
        "posts_youtube.json": {
            "csv_file": "posts Data Dump - Youtube.csv",
            "text_columns": ["title", "description", "content"]
        },
        "comments_reddit.json": {
            "csv_file": "comments Data Dump - Reddit.csv",
            "text_columns": ["body", "content", "text"]
        },
        "comments_youtube.json": {
            "csv_file": "comments Data Dump - Youtube.csv",
            "text_columns": ["text", "content", "comment"]
        }
    }
    
    conversion_results = {}
    
    for json_filename, config in files_config.items():
        csv_filename = config["csv_file"]
        text_columns = config["text_columns"]
        
        csv_path = os.path.join(DATA_DIR, csv_filename)
        json_path = os.path.join(JSON_DIR, json_filename)
        final_path = os.path.join(FINAL_DIR, json_filename)
        embedded_path = os.path.join(EMBEDDED_JSON_DIR, json_filename)  # New path for embedded data
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            original_rows = len(df)
            logger.info(f"Processing {csv_filename} with {original_rows} rows")
            
            # Process data with new embeddings
            processed_df = process_data_with_embeddings(df, text_columns)
            
            # Convert to JSON
            json_data = processed_df.to_dict(orient="records")
            
            # Save to all directories including the new embedded_json_data directory
            for path in [json_path, final_path, embedded_path]:
                with open(path, "w", encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # Verify all data is preserved
            final_rows = len(json_data)
            if original_rows == final_rows:
                conversion_results[csv_filename] = f"Success ({original_rows} rows with embeddings)"
            else:
                conversion_results[csv_filename] = f"Warning: {original_rows} -> {final_rows} rows"
                
        except FileNotFoundError:
            conversion_results[csv_filename] = "Failed (File not found)"
        except Exception as e:
            conversion_results[csv_filename] = f"Failed (Error: {str(e)})"
    
    return {
        "message": "Conversion process completed with embeddings.",
        "results": conversion_results
    }

@router.get("/generate-embeddings-only")
async def generate_embeddings_only():
    """
    Only generate new embeddings for existing JSON files without converting CSV again
    """
    json_files = [
        "posts_reddit.json",
        "posts_youtube.json",
        "comments_reddit.json",
        "comments_youtube.json"
    ]
    
    results = {}
    
    for json_filename in json_files:
        json_path = os.path.join(JSON_DIR, json_filename)
        final_path = os.path.join(FINAL_DIR, json_filename)
        embedded_path = os.path.join(EMBEDDED_JSON_DIR, json_filename)  # New path for embedded data
        
        try:
            # Load existing JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert back to DataFrame for processing
            df = pd.DataFrame(data)
            original_rows = len(df)
            
            # Process with new embeddings
            processed_df = process_data_with_embeddings(df)
            
            # Convert back to JSON
            updated_data = processed_df.to_dict(orient="records")
            
            # Save to final directory and embedded_json_data directory
            for path in [final_path, embedded_path]:
                with open(path, "w", encoding='utf-8') as f:
                    json.dump(updated_data, f, indent=2, ensure_ascii=False)
            
            results[json_filename] = f"Success ({original_rows} rows updated)"
            
        except Exception as e:
            results[json_filename] = f"Failed (Error: {str(e)})"
    
    return {
        "message": "Embedding generation completed.",
        "results": results
    }

# Helper function to read CSV and drop embeddings
def load_csv_without_embeddings(file_path: str):
    df = pd.read_csv(file_path)
    columns_to_drop = ['embeddings', 'embeddings_vec', 'small_embedding']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    return df.to_dict(orient="records")

# Original endpoints preserved
@router.get("/posts/reddit")
async def get_reddit_posts():
    file_path = os.path.join(DATA_DIR, "posts Data Dump - Reddit.csv")
    try:
        return load_csv_without_embeddings(file_path)
    except FileNotFoundError:
        return {"error": "File not found."}

@router.get("/posts/youtube")
async def get_youtube_posts():
    file_path = os.path.join(DATA_DIR, "posts Data Dump - Youtube.csv")
    try:
        return load_csv_without_embeddings(file_path)
    except FileNotFoundError:
        return {"error": "File not found."}

@router.get("/comments/reddit")
async def get_reddit_comments():
    file_path = os.path.join(DATA_DIR, "comments Data Dump - Reddit.csv")
    try:
        return load_csv_without_embeddings(file_path)
    except FileNotFoundError:
        return {"error": "File not found."}

@router.get("/comments/youtube")
async def get_youtube_comments():
    file_path = os.path.join(DATA_DIR, "comments Data Dump - Youtube.csv")
    try:
        return load_csv_without_embeddings(file_path)
    except FileNotFoundError:
        return {"error": "File not found."}