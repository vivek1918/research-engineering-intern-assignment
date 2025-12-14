#!/usr/bin/env python3
"""
Script to convert CSV files to JSON format.
Converts 4 CSV files (Reddit posts/comments, YouTube posts/comments) to JSON.
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Dict, Any


def convert_csv_to_json(csv_path: str, json_path: str, file_name: str) -> int:
    """
    Convert a CSV file to JSON format.
    
    Args:
        csv_path: Path to the CSV file
        json_path: Path where JSON file will be saved
        file_name: Name of the output JSON file
        
    Returns:
        Number of records converted
    """
    try:
        print(f"\n📖 Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Clean up the dataframe
        df = df.where(pd.notna(df), None)
        
        # Convert to list of dictionaries
        records = df.to_dict(orient='records')
        
        # Create output path
        output_file = Path(json_path) / file_name
        
        # Write to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Successfully converted {len(records)} records")
        print(f"💾 Saved to: {output_file}")
        
        return len(records)
        
    except Exception as e:
        print(f"❌ Error converting {csv_path}: {e}")
        return 0


def main():
    """Main function to orchestrate CSV to JSON conversion."""
    
    # Define paths
    backend_dir = Path(__file__).parent.parent
    data_dir = backend_dir / "data"
    json_output_dir = backend_dir / "json_data"
    
    # Create json_data directory if it doesn't exist
    json_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {json_output_dir}")
    
    # Define CSV files to convert
    csv_files = [
        ("reddit_posts.csv", "posts_reddit.json"),
        ("reddit_comments.csv", "comments_reddit.json"),
        ("youtube_posts.csv", "posts_youtube.json"),
        ("youtube_comments.csv", "comments_youtube.json"),
    ]
    
    total_records = 0
    successful_conversions = 0
    
    print("\n🚀 Starting CSV to JSON conversion process...\n")
    print("=" * 60)
    
    for csv_name, json_name in csv_files:
        csv_file_path = data_dir / csv_name
        
        if not csv_file_path.exists():
            print(f"\n⚠️  File not found: {csv_file_path}")
            continue
        
        records = convert_csv_to_json(str(csv_file_path), str(json_output_dir), json_name)
        if records > 0:
            total_records += records
            successful_conversions += 1
    
    print("\n" + "=" * 60)
    print(f"\n📊 Conversion Summary:")
    print(f"   ✅ Successfully converted: {successful_conversions}/4 files")
    print(f"   📝 Total records processed: {total_records}")
    print(f"   📂 Output directory: {json_output_dir}")
    print("\n✨ Conversion complete!\n")


if __name__ == "__main__":
    main()
