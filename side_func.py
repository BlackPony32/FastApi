import pandas as pd
import os
import streamlit as st  # Make sure to import streamlit for error logging

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
last_uploaded_file_path = os.path.join(UPLOAD_DIR, "last_uploaded.csv")
def identify_file(uploaded_file):
    try:
        df = pd.read_csv(last_uploaded_file_path, encoding='utf-8')
        columns = set(df.columns)

        # Define known column sets
        representative_details_cols = {'Role', 'Id', 'Status', 'Name', 'Email', 'Phone number', 'Assigned customers',
                                       'Active customers', 'Inactive customers', 'Total visits', 'Total photos',
                                       'Total notes', 'Total working hours', 'Total break hours', 'Total travel distance'}

        order_sales_summary_cols = {'Id', 'Created at', 'Created by', 'Customer', 'Representative', 'Grand total',
                                    'Balance', 'Paid', 'Delivery status', 'Payment status', 'Order Status',
                                    'Delivery methods', 'Manufacturer name', 'Product name', 'QTY', 'Delivered',
                                    'Item specific discount', 'Manufacturer specific discount', 'Total invoice discount',
                                    'Discount type', 'Customer discount', 'Free cases'}

        reps_summary_cols = {'Role', 'Id', 'Name', 'Date', 'Start day', 'End day', 'Total time', 'Break',
                             'Travel distance', 'Visits', 'Photos', 'Notes', 'Orders', 'New clients',
                             'Cases sold', 'Total revenue'}

        best_sellers_cols = {'Product name', 'Manufacturer name', 'Category name', 'SKU', 'Available cases (QTY)',
                             'Wholesale price', 'Retail price', 'Cases sold', 'Total revenue'}

        skus_not_ordered_cols = {'Product name', 'Manufacturer name', 'Category name', 'SKU', 'Available cases (QTY)',
                                 'Wholesale price', 'Retail price'}

        third_party_sales_summary_cols = {'Id', 'Created at', 'Created by', 'Customer', 'Representative', 'Grand total',
                                          'Manufacturer name', 'Product name', 'QTY', 'Item specific discount',
                                          'Manufacturer specific discount', 'Total invoice discount', 'Discount type',
                                          'Customer discount', 'Free cases'}

        top_customers_cols = {'Name', 'Group', 'Billing address', 'Billing city', 'Billing state',
                              'Billing zip', 'Shipping address', 'Shipping city', 'Shipping state',
                              'Shipping zip', 'Phone', 'Payment terms', 'Customer discount', 'Territory',
                              'Website', 'Tags', 'Contact name', 'Contact role', 'Contact phone',
                              'Contact email', 'Order direct access', 'Total orders', 'Total sales',
                              'Business Fax', 'Primary payment method', 'Licenses & certifications'}

        # Identify file type based on columns
        if columns == representative_details_cols:
            return "Representative Details report"

        elif columns == order_sales_summary_cols:
            return "Order Sales Summary report"

        elif columns == reps_summary_cols:
            return "Reps Summary report"

        elif columns == best_sellers_cols:
            if df['Cases sold'].any():
                return "Best Sellers report"
            elif "SKU" in file_name:
                return "SKU's Not Ordered report"
            else:
                return "Unknown (similar columns to Best Sellers and SKU's Not Ordered)"

        elif columns == skus_not_ordered_cols:
            if "Low" in file_name:
                return "Low Stock Inventory report"
            elif "Current" in file_name:
                return "Current Inventory report"
            else:
                return "Unknown (similar columns to Low Stock and Current Inventory)"

        elif columns == third_party_sales_summary_cols:
            return "3rd Party Sales Summary report"

        elif columns == top_customers_cols and "Top" in file_name:
            return "Top Customers report"

        elif columns == top_customers_cols:
            return "Customer Details report"

        else:
            return "Unknown"

    except Exception as e:
        # Log the exception for debugging
        st.error(f"Error reading file: {e}")
        return "Invalid File"
