#!/usr/bin/env python3
"""
combine_datasets.py

Transform results_abs_edu.csv to match ABS dataset format and combine with abs-datasets.csv
"""

import pandas as pd
import hashlib
import re
from urllib.parse import urlparse
from datetime import datetime

def generate_id_from_url(url):
    """Generate a unique ID from the file URL"""
    # Extract meaningful parts from URL
    parsed = urlparse(url)
    path_parts = parsed.path.split('/')
    
    # Try to find meaningful identifiers in the URL
    meaningful_parts = []
    for part in path_parts:
        if part and part not in ['download', 'document', 'pdf', 'xlsx', 'csv']:
            meaningful_parts.append(part)
    
    if meaningful_parts:
        # Use the most specific part (usually the last meaningful one)
        base_id = meaningful_parts[-1].upper().replace('-', '_')
        # Clean up the ID
        base_id = re.sub(r'[^A-Z0-9_]', '_', base_id)
        return f"EDU_{base_id}"
    else:
        # Fallback to hash if no meaningful parts found
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8].upper()
        return f"EDU_{url_hash}"

def extract_frequency_from_title(title, description):
    """Try to infer frequency from title and description"""
    text = f"{title} {description}".lower()
    
    if any(word in text for word in ['annual', 'yearly', 'year']):
        return 'annual'
    elif any(word in text for word in ['quarterly', 'quarter']):
        return 'quarterly'
    elif any(word in text for word in ['monthly', 'month']):
        return 'monthly'
    elif any(word in text for word in ['weekly', 'week']):
        return 'weekly'
    elif any(word in text for word in ['daily', 'day']):
        return 'daily'
    else:
        return ''

def transform_education_data(input_file):
    """Transform education CSV to match ABS format"""
    df = pd.read_csv(input_file)
    
    # Create new columns matching ABS format
    transformed_data = []
    
    for _, row in df.iterrows():
        # Generate ID from URL
        dataset_id = generate_id_from_url(row['file_url'])
        
        # Use page_title as title, fallback to anchor_text
        title = row['page_title'] if pd.notna(row['page_title']) else row['anchor_text']
        
        # Use page_description as description
        description = row['page_description'] if pd.notna(row['page_description']) else ''
        
        # Set agency as Department of Education
        agency = 'Department of Education'
        
        # Use data_collected_date if available
        collected = row['data_collected_date'] if pd.notna(row['data_collected_date']) else ''
        
        # Try to infer frequency
        freq = extract_frequency_from_title(title, description)
        
        # Use file_url as both api_url and download_url
        api_url = row['file_url']
        download_url = row['file_url']
        
        # Create tags from available fields
        tags = []
        if pd.notna(row['page_tags']):
            tags.append(row['page_tags'])
        if pd.notna(row['file_ext']):
            tags.append(f"format_{row['file_ext'].replace('.', '')}")
        if pd.notna(row['content_type']):
            content_type_tag = row['content_type'].replace('/', '_').replace('-', '_')
            tags.append(f"content_type_{content_type_tag}")
        
        tags_str = ';'.join(tags)
        
        transformed_data.append({
            'id': dataset_id,
            'title': title,
            'description': description,
            'agency': agency,
            'collected': collected,
            'freq': freq,
            'api_url': api_url,
            'download_url': download_url,
            'tags': tags_str
        })
    
    return pd.DataFrame(transformed_data)

def main():
    print("Transforming education data...")
    edu_df = transform_education_data('results_abs_edu.csv')
    print(f"Transformed {len(edu_df)} education records")
    
    print("Loading ABS data...")
    abs_df = pd.read_csv('abs-datasets.csv')
    print(f"Loaded {len(abs_df)} ABS records")
    
    print("Combining datasets...")
    combined_df = pd.concat([abs_df, edu_df], ignore_index=True)
    
    # Ensure all columns are present and in the right order
    columns = ['id', 'title', 'description', 'agency', 'collected', 'freq', 'api_url', 'download_url', 'tags']
    combined_df = combined_df[columns]
    
    # Save combined dataset
    output_file = 'combined_datasets.csv'
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Combined dataset saved to {output_file}")
    print(f"Total records: {len(combined_df)}")
    print(f"- ABS records: {len(abs_df)}")
    print(f"- Education records: {len(edu_df)}")
    
    # Show sample of education records
    print("\nSample of transformed education records:")
    print(edu_df.head(3).to_string())

if __name__ == "__main__":
    main()
