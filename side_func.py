import pandas as pd
import os
import streamlit as st  # Make sure to import streamlit for error logging
import re
#from main import file_name

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def get_file_name():
    file_name = [
        f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))
    ]
    try:
        return file_name[0]
    except Exception as e:
        return "Invalid File"

def identify_file(uploaded_file):
    try:
        file_name = get_file_name()
        last_uploaded_file_path = os.path.join(UPLOAD_DIR, file_name)
        df = pd.read_csv(last_uploaded_file_path, encoding='utf-8')
        columns = set(df.columns)

        
        # Identify file type based on columns
        if file_name == 'third_party_sales_summary.csv':
            return "3rd Party Sales Summary report"
        elif file_name == 'order_sales_summary.csv':
            return "Order Sales Summary report"
        elif file_name == 'best_sellers.csv':
            return "Best Sellers report"
        elif file_name == 'rep_details.csv':
            return "Representative Details report"
        elif file_name == 'reps_summary.csv':
            return "Reps Summary report" #"Unknown (similar columns to Low Stock and Current Inventory)"
        elif file_name == 'sku_not_ordered.csv':
            return "SKU's Not Ordered report"
        elif file_name == 'low_stock_inventory.csv':
            return "Low Stock Inventory report"
        elif file_name == 'current_inventory.csv':
            return "Current Inventory report"
        elif file_name == 'top_customers.csv':
            return "Top Customers report"
        elif file_name == 'customer_details.csv':
            return "Customer Details report"
        else:
            return "Unknown"
        


    except Exception as e:
        # Log the exception for debugging
        st.error(f"Error reading file: {e}")
        return "Invalid File"

def extract_filename(url):
    # Extract the filename with extension from the URL
    filename_with_extension = url.split("/")[-1]
    
    # Remove the extension
    filename_without_extension = filename_with_extension.rsplit(".", 1)[0]

    clean_filename = re.sub(r'(-\d{2}[A-Z]{3}\d{2}(-\d{2}[A-Z]{3}\d{2})?)$', '', filename_without_extension)
    
    return clean_filename

def get_csv_columns(last_uploaded_file_path):
    """
    This function takes the path to a CSV file and returns a list of its column names.
    
    :param last_uploaded_file_path: Path to the CSV file.
    :return: List of column names.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(last_uploaded_file_path)
        
        # Get the list of columns
        columns = df.columns.tolist()
        
        return columns
    except FileNotFoundError:
        return "File not found. Please check the path."
    except pd.errors.EmptyDataError:
        return "No data. The file is empty."
    except pd.errors.ParserError:
        return "Error parsing data. Please check the file format."
    except Exception as e:
        return f"An error occurred: {e}"