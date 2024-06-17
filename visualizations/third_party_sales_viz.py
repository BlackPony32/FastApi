import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# from utils import plotly_preproc

#is used to call instead of st.plotly_chart(fig)
def plotly_preproc(fig):
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        st.plotly_chart(fig, theme=None, use_container_width=True)

def preprocess_data(data):
    """
    Data preprocessing: data type conversion and cleaning.

    Args:
        data: A Pandas DataFrame with the source data.

    Returns:
        Pandas DataFrame with the processed data.
    """

    # Identify numeric columns automatically
    numeric_cols = data.select_dtypes(include=np.number).columns

    # Process numeric columns
    for col in numeric_cols:
        # Check for missing values (NaN)
        if np.isnan(data[col]).any():
            # Fill missing values with 0 (you can choose another strategy)
            data[col].fillna(0, inplace=True)
            print(f"Warning: Column '{col}' contains missing values (NaN). Filled with 0.")

    # Remove currency symbols and thousands separators
    data[numeric_cols] = data[numeric_cols].replace('[$,]', '', regex=True).astype(float)

    return data

#Total sales
def visualize_product_analysis(data, product_col='Product name', grand_total_col='Grand total'):
    product_data = data.groupby(product_col)[grand_total_col].agg(['sum', 'count']).sort_values(by='sum', ascending=False)

    # Create two tabs for the visualizations
    tab1, tab2 = st.tabs(["Total Sales", "Order Distribution"])

    with tab1:
        # fig = px.bar(product_data, x=product_data.index, y='sum', title="Total Sales by Product",
        #              color='sum', color_continuous_scale='Viridis')
        # fig.update_layout(xaxis_tickangle=45, yaxis_title="Sales Amount", xaxis_title="Product")

        fig = px.pie(product_data, values='count', names=product_data.index,
                                        title='Order Distribution')
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig)
        st.markdown("""
## Total Sales by Category

This chart shows how total sales are split across product categories over time. See which categories drive sales in each period and spot any trends.
""")

    with tab2:
        fig = px.bar(product_data, x=product_data.index, y='count', title="Distribution of Orders by Product",
                     color='count', color_continuous_scale='Cividis') 
        fig.update_layout(xaxis_tickangle=45, yaxis_title="Number of Orders", xaxis_title="Product")
        st.plotly_chart(fig)
        st.markdown("""
## Order Distribution by Product

This chart displays the number of orders for each product, indicating their popularity and helping you manage inventory effectively.
""")

#Sales amount for each client (top 10)
def visualize_sales_trends(data, customer_col='Customer', product_col='Product name',
                           grand_total_col='Grand total', qty_col='QTY'):
    top_customers = data.groupby(customer_col)[grand_total_col].sum().nlargest(10)
    fig = px.bar(top_customers, 
                  x=top_customers.index, 
                  y=top_customers.values,
                  title="Top 10 Customers by Sales Amount", 
                  color=top_customers.values,
                  color_continuous_scale='Bluyl')
    fig.update_layout(xaxis_tickangle=45, 
                      yaxis_title="Sales Amount", 
                      xaxis_title="Customer",
                      yaxis_autorange="reversed") # Reverse the y-axis
    st.plotly_chart(fig)
    st.markdown("""
    ## Top Customer Insights
    This chart highlights your top 10 customers by sales revenue. Prioritize these key relationships to drive future sales and consider loyalty programs to encourage repeat business.
    """)
        

    

#Analysis of discounts
def visualize_discount_analysis(data, discount_type_col='Discount type', 
                               total_discount_col='Total invoice discount'):
    tab1, tab2, tab3 = st.tabs(["By Type", "Top Customers", "Distribution"])

    with tab1:
        discount_amounts = data.groupby(discount_type_col)[total_discount_col].sum().sort_values(ascending=False)
        fig = px.scatter(discount_amounts, 
                     x=discount_amounts.index, 
                     y=discount_amounts.values, 
                     title="Discount Amount by Type",
                     size=discount_amounts.values, 
                     color=discount_amounts.index,
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     symbol=discount_amounts.index)  # Use different symbols for each type

        fig.update_traces(marker=dict(size=20,  # Adjust marker size for better visibility
                                  line=dict(width=2, color='DarkSlateGrey')), # Add outline to markers
                      selector=dict(mode='markers'))
        fig.update_layout(xaxis_tickangle=45, 
                      yaxis_title="Discount Amount", 
                      xaxis_title="Discount Type", 
                      plot_bgcolor='white') # White background
        st.plotly_chart(fig)
        st.markdown("""
## Discount Type Breakdown

This chart shows the total discount amount for each type. Remember to verify data accuracy and consider the impact of different discount types on profitability before making changes to your discount strategy.
""")

    with tab2:
        top_customers_discount = data.groupby('Customer')[total_discount_col].sum().nlargest(10)
        fig = px.bar(top_customers_discount, 
                     x=top_customers_discount.index, 
                     y=top_customers_discount.values,
                     title="Discount Amount by Customer (Top 10)", 
                     color=top_customers_discount.index,  # Color by customer
                     color_discrete_sequence=px.colors.qualitative.Bold)  
        fig.update_layout(xaxis_tickangle=45, 
                          yaxis_title="Discount Amount", 
                          xaxis_title="Customer",
                          xaxis={'categoryorder': 'total descending'}) # Order by discount amount
        st.plotly_chart(fig)
        st.markdown("""
## Top Discount Recipients 

This chart reveals the customers receiving the highest total discounts. Ensure these discounts align with your customer relationship strategies and consider other incentive programs you might offer.
""")

    with tab3:
        fig = px.box(data, 
                      x=discount_type_col, 
                      y=total_discount_col, 
                      title="Distribution of Discount Amount by Type",
                      color=discount_type_col,
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(xaxis_tickangle=45, 
                          yaxis_title="Discount Amount", 
                          xaxis_title="Discount Type")
        st.plotly_chart(fig)
        st.markdown("""
## Discount Distribution Analysis

This plot displays the distribution of discount amounts for each type. It's essential to verify data accuracy and assess the overall impact of discounts on your profit margins.
""")

#Plot with coloring of points by product type
def visualize_combined_analysis(data, product_col='Product name',
                               grand_total_col='Grand total', qty_col='QTY',
                               delivery_status_col='Delivery status'):
    fig = px.bar(data, x=qty_col, y=grand_total_col, color=product_col,
                     title="Dependence between Quantity and Amount (by Product)",
                     labels={"x": "Quantity", "y": "Sales Amount"})

    # fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig)
    st.markdown("""
        ## Sales by Quantity and Product

This chart shows how sales revenue varies with the quantity of products sold, broken down by product name. Analyze which products contribute the most at different quantity levels.""")
#Analyzes discounts
def analyze_discounts(data):
    discount_counts = data["Discount type"].value_counts()
    fig = px.pie(discount_counts, 
                  values=discount_counts.values, 
                  names=discount_counts.index,
                  title="Distribution of Discount Types",
                  hole=0.4,  # Create a hole for the donut chart
                  color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)
    st.markdown("""
## Discount Usage

This chart presents the distribution of discount types used in orders.  It highlights the proportion of orders associated with each discount category.
""")

def area_visualisation(data):
    columns = ['Grand total', 'Manufacturer specific discount', 'Customer discount']
    fig = px.area(data[columns])
    st.plotly_chart(fig)
    st.markdown("""
        ## Sales, Manufacturer, and Customer Discounts Over Time

This area chart displays how the grand total, manufacturer discounts, and customer discounts have fluctuated over time. Track how these values change, identifying periods of high discounts and understanding the overall impact of discounts on sales revenue. 
        """)