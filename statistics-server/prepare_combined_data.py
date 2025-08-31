"""
Prepare combined dataset from datasets.csv and results_abs_edu.csv
Transform education data to match the datasets.csv format and combine them.
"""

import pandas as pd
import os
import hashlib

def generate_unique_id(title, url):
    """Generate a unique ID from title and URL."""
    # Create a hash from title and URL to ensure uniqueness
    content = f"{title}_{url}"
    return "EDU_" + hashlib.md5(content.encode()).hexdigest()[:12].upper()

def transform_education_data():
    """Transform education data to match datasets.csv format."""
    # Read the education data
    edu_path = "../results_abs_edu.csv"
    if not os.path.exists(edu_path):
        print(f"Error: {edu_path} not found")
        return None
    
    edu_df = pd.read_csv(edu_path)
    print(f"Loaded {len(edu_df)} education records")
    
    # Transform to match datasets.csv format
    transformed_records = []
    
    for _, row in edu_df.iterrows():
        # Extract meaningful information
        title = row.get('page_title', '').replace(' - Department of Education, Australian Government', '')
        description = row.get('page_description', '')
        url = row.get('file_url', '')
        
        # Skip if missing essential data
        if not title or not description or not url:
            continue
        
        # Generate unique ID
        dataset_id = generate_unique_id(title, url)
        
        # Create record matching datasets.csv format
        record = {
            'id': dataset_id,
            'title': title.strip(),
            'description': description.strip(),
            'agency': 'Department of Education',
            'collected': '',  # Not available in education data
            'freq': '',       # Not available in education data
            'api_url': '',    # Not available in education data
            'download_url': url,
            'tags': 'education;government;australia',
            'url': url  # Use the file URL as the source URL
        }
        
        transformed_records.append(record)
    
    # Create DataFrame
    edu_transformed_df = pd.DataFrame(transformed_records)
    print(f"Transformed {len(edu_transformed_df)} education records")
    
    return edu_transformed_df

def combine_datasets():
    """Combine existing datasets.csv with transformed education data."""
    # Read existing datasets
    datasets_df = pd.read_csv("datasets.csv")
    print(f"Loaded {len(datasets_df)} existing datasets")
    
    # Transform education data
    edu_df = transform_education_data()
    if edu_df is None:
        return False
    
    # Ensure both DataFrames have the same columns
    required_cols = ['id', 'title', 'description', 'agency', 'collected', 'freq', 'api_url', 'download_url', 'tags', 'url']
    
    # Add missing columns to datasets_df if needed
    for col in required_cols:
        if col not in datasets_df.columns:
            datasets_df[col] = ''
    
    # Add missing columns to edu_df if needed  
    for col in required_cols:
        if col not in edu_df.columns:
            edu_df[col] = ''
    
    # Reorder columns to match
    datasets_df = datasets_df[required_cols]
    edu_df = edu_df[required_cols]
    
    # Combine the DataFrames
    combined_df = pd.concat([datasets_df, edu_df], ignore_index=True)
    
    # Save combined dataset
    combined_df.to_csv("combined_datasets.csv", index=False)
    print(f"✅ Successfully created combined_datasets.csv with {len(combined_df)} total records")
    print(f"   - Original datasets: {len(datasets_df)}")
    print(f"   - Education datasets: {len(edu_df)}")
    
    # Show sample of new education entries
    print("\nSample education entries:")
    sample_edu = edu_df.head(3)[['id', 'title', 'agency']]
    print(sample_edu.to_string(index=False))
    
    return True

if __name__ == "__main__":
    combine_datasets()
