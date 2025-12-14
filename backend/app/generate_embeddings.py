#!/usr/bin/env python3
"""
Script to generate embeddings from JSON files and save to embedded_json_data directory.
Clears existing files and creates new embedded JSON files.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def generate_small_embedding(text: str, embedding_model, max_tokens: int = 128) -> list:
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


def combine_text_fields(record: Dict[str, Any], text_columns: List[str]) -> str:
    """Combine multiple text fields for better embedding context"""
    combined_text = ""
    for col in text_columns:
        if col in record and record[col] is not None:
            combined_text += f" {str(record[col])}"
    return combined_text.strip()


def process_json_with_embeddings(
    json_path: str, 
    output_path: str, 
    embedding_model, 
    text_columns: List[str] = None
) -> int:
    """
    Process JSON file and add embeddings to each record.
    
    Args:
        json_path: Path to input JSON file
        output_path: Path to save output JSON file
        embedding_model: SentenceTransformer model instance
        text_columns: List of column names to use for embedding
        
    Returns:
        Number of records processed
    """
    try:
        logger.info(f"📖 Reading JSON file: {json_path}")
        
        # Read JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        original_count = len(data)
        logger.info(f"   Loaded {original_count} records")
        
        # Auto-detect text columns if not specified
        if text_columns is None or len(text_columns) == 0:
            text_columns = []
            if data:
                sample_record = data[0]
                for key, value in sample_record.items():
                    if isinstance(value, str) and len(value) > 5:
                        text_columns.append(key)
            logger.info(f"   Auto-detected text columns: {text_columns}")
        
        # Process records and add embeddings
        logger.info(f"🧠 Generating embeddings for {original_count} records...")
        processed_data = []
        
        for idx, record in enumerate(data):
            if (idx + 1) % 100 == 0:
                logger.info(f"   Processing record {idx + 1}/{original_count}...")
            
            # Combine text fields
            combined_text = combine_text_fields(record, text_columns)
            
            # Generate embedding
            embedding = generate_small_embedding(combined_text, embedding_model, max_tokens=150)
            
            # Add embedding to record
            record['small_embedding'] = embedding
            processed_data.append(record)
        
        # Save to output file
        logger.info(f"💾 Saving {len(processed_data)} records to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ Successfully processed {len(processed_data)} records")
        return len(processed_data)
        
    except Exception as e:
        logger.error(f"❌ Error processing {json_path}: {e}")
        return 0


def clear_directory(directory_path: str) -> int:
    """Clear all files from a directory"""
    try:
        path = Path(directory_path)
        if not path.exists():
            logger.info(f"📁 Creating directory: {directory_path}")
            path.mkdir(parents=True, exist_ok=True)
            return 0
        
        files_removed = 0
        for file in path.glob('*.json'):
            file.unlink()
            logger.info(f"🗑️  Removed: {file.name}")
            files_removed += 1
        
        logger.info(f"✅ Cleared {files_removed} files from {directory_path}")
        return files_removed
        
    except Exception as e:
        logger.error(f"❌ Error clearing directory {directory_path}: {e}")
        return 0


def main():
    """Main function to generate embeddings from JSON files"""
    
    # Define paths
    backend_dir = Path(__file__).parent.parent
    json_data_dir = backend_dir / "json_data"
    embedded_data_dir = backend_dir / "embedded_json_data"
    
    logger.info("=" * 70)
    logger.info("🚀 Starting JSON to Embedded JSON conversion process...")
    logger.info("=" * 70)
    
    # Step 1: Clear existing embedded_json_data directory
    logger.info(f"\n📂 Clearing embedded_json_data directory: {embedded_data_dir}")
    cleared = clear_directory(str(embedded_data_dir))
    logger.info(f"   Removed {cleared} existing files\n")
    
    # Step 2: Initialize embedding model
    logger.info("🤖 Loading SentenceTransformer model: all-MiniLM-L6-v2")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("   ✅ Model loaded successfully\n")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return
    
    # Step 3: Define files to process with their text columns
    files_config = [
        {
            "input": "posts_reddit.json",
            "output": "posts_reddit.json",
            "text_columns": ["title", "body", "content", "text"]
        },
        {
            "input": "posts_youtube.json",
            "output": "posts_youtube.json",
            "text_columns": ["title", "description", "content", "text"]
        },
        {
            "input": "comments_reddit.json",
            "output": "comments_reddit.json",
            "text_columns": ["body", "content", "text", "comment"]
        },
        {
            "input": "comments_youtube.json",
            "output": "comments_youtube.json",
            "text_columns": ["text", "content", "comment", "body"]
        }
    ]
    
    # Step 4: Process each JSON file
    logger.info(f"📝 Processing JSON files from: {json_data_dir}\n")
    
    total_records = 0
    successful_conversions = 0
    
    for file_config in files_config:
        input_file = json_data_dir / file_config["input"]
        output_file = embedded_data_dir / file_config["output"]
        
        if not input_file.exists():
            logger.warning(f"\n⚠️  File not found: {input_file}")
            continue
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Processing: {file_config['input']}")
        logger.info(f"{'=' * 70}")
        
        records = process_json_with_embeddings(
            str(input_file),
            str(output_file),
            embedding_model,
            file_config["text_columns"]
        )
        
        if records > 0:
            total_records += records
            successful_conversions += 1
    
    # Step 5: Summary
    logger.info(f"\n{'=' * 70}")
    logger.info("📊 Conversion Summary:")
    logger.info(f"{'=' * 70}")
    logger.info(f"   ✅ Successfully converted: {successful_conversions}/4 files")
    logger.info(f"   📝 Total records processed: {total_records:,}")
    logger.info(f"   📂 Output directory: {embedded_data_dir}")
    logger.info(f"\n✨ Embedding generation complete!\n")


if __name__ == "__main__":
    main()
