import pandas as pd
import os

def remove_columns_from_csv(input_csv, output_csv, columns_to_remove):
    """
    Reads a CSV file, removes specified columns, and saves the result to a new CSV file.

    Args:
        input_csv (str): The path to the input CSV file.
        output_csv (str): The path where the new CSV file will be saved.
        columns_to_remove (list): A list of column names to be dropped.
    """
    try:
        # Check if the input file exists
        if not os.path.exists(input_csv):
            print(f"Error: The input file '{input_csv}' was not found.")
            return

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(input_csv)
        print(f"Original DataFrame has columns: {df.columns.tolist()}")

        # Drop the specified columns.
        # The `errors='ignore'` argument prevents the code from failing
        # if a column in the list does not exist in the DataFrame.
        df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')
        print(f"Cleaned DataFrame has columns: {df_cleaned.columns.tolist()}")

        # Save the updated DataFrame to a new CSV file
        df_cleaned.to_csv(output_csv, index=False)

        print(f"Successfully saved the new CSV file to '{output_csv}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    dummy_df = pd.read_csv('datasets.csv')
    dummy_df.to_csv('sample_data.csv', index=False)
    print("Created 'sample_data.csv' for demonstration.\n")

    # Define the file paths and the columns to be removed
    input_file = 'sample_data.csv'
    output_file = 'modified_data.csv'
    columns_to_drop = ['collected', 'freq','api_url','download_url']

    # Call the function to remove the columns
    remove_columns_from_csv(input_file, output_file, columns_to_drop)
