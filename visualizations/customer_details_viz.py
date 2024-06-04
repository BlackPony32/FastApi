import plotly.express as px
import pandas as pd
import streamlit as st
import numpy as np
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
#Visualization Customer_details
def plot_orders_and_sales_plotly(df, group_col='Group'):
    orders = df.groupby(group_col)["Total orders"].sum()
    sales = df.groupby(group_col)["Total sales"].sum()

    fig = px.bar(x=orders.index, y=[orders, sales], barmode="group",
                 labels={"x": "Customer Group", "value": "Count/Amount"},  # Remove 'variable' label
                 title="Comparison of Total Orders and Sales by Customer Group")

    fig.update_traces(marker_color=["skyblue", "coral"],
                      marker_line_color=["steelblue", "darkred"],
                      marker_line_width=1,
                      texttemplate='%{y:.2s}', textposition='outside',
                      name="Total Orders",  # Set custom names for traces
                      legendgroup="group1",  # Assign traces to the same legend group
                      legendgrouptitle_text="Metrics")  # Set legend group title

    fig.update_layout(xaxis_tickangle=45,
                      xaxis_title_font_size=12,
                      yaxis_title_font_size=12,
                      showlegend=False)  # Hide the legend

    st.plotly_chart(fig)
    st.markdown("""
## Customer Group Performance: Orders & Sales

These visualizations compare total orders and sales for each customer group, revealing valuable insights about your most important customer segments. Use this data to identify high-value customers, understand purchasing behavior, and optimize your sales and marketing strategies.
""")
#________________________________________________________________
def bar_plot_sorted_with_percentages(df, col='Payment terms'):
    counts = df[col].value_counts().sort_values(ascending=False)
    percentages = (counts / len(df)) * 100
    df_plot = pd.DataFrame({'Category': counts.index, 'Count': counts.values, 'Percentage': percentages})

    fig = px.bar(df_plot, x='Category', y='Count', text='Percentage', title=f"Distribution of clients by {col}",
                 hover_data=['Count'])  # Add hover data for precise count values
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', title_x=0.5,
                      xaxis_tickangle=45, xaxis_tickfont_size=10)  # Customize axis ticks
    st.plotly_chart(fig)
    st.markdown("""
## Payment Terms: Understanding Client Preferences

This visualization shows how your clients are distributed across different payment terms.  Use these insights to optimize your cash flow management, tailor your offerings to different customer segments, and make informed decisions about credit risk.
""")
#Data distribution visualization function
def create_interactive_non_zero_sales_plot(df, sales_col='Total sales', threshold=500):
    df_filtered = df[df[sales_col] > 0]
    df_below_threshold = df_filtered[df_filtered[sales_col] <= threshold]
    df_above_threshold = df_filtered[df_filtered[sales_col] > threshold]
    counts_below = df_below_threshold[sales_col].value_counts().sort_index()
    count_above = df_above_threshold[sales_col].count()
    values = counts_below.index.tolist() + [f"{threshold}+"]
    counts = counts_below.values.tolist() + [count_above]
    df_plot = pd.DataFrame({'Sales Value': values, 'Count': counts})

    fig = px.line(df_plot, x='Sales Value', y='Count', markers=True, 
                  title=f"Distribution of non-zero {sales_col}", 
                  hover_data={'Count': True})  # Add hover data for precise count
    fig.update_layout(title_x=0.5, xaxis_title="Value of total sales", 
                      yaxis_title="Number of entries")
    fig.update_traces(line_color='blue')
    st.plotly_chart(fig)
    st.markdown("""
## Sales Value Distribution: Insights for Growth

This visualization reveals how non-zero total sales values are distributed. Use these insights to refine your pricing and promotion strategies, enhance sales forecasting, optimize product development, and effectively manage potential risks.
""")
#Average total sales by customer group and billing state
def create_interactive_average_sales_heatmap(df):
    df['Total sales'] = df['Total sales'].apply(pd.to_numeric, errors='coerce')  # Convert to numeric
    average_sales = df.groupby(["Group", "Billing state"])["Total sales"].mean().unstack()

    fig = px.imshow(average_sales, color_continuous_scale='YlGnBu', 
                    title="Average Total Sales by Customer Group and State")
    fig.update_layout(xaxis_title="Billing State", yaxis_title="Customer Group")
    fig.update_xaxes(side="top")  # Move x-axis labels to the top
    st.plotly_chart(fig)
    st.markdown("""
## Sales Performance: Customer Groups & Geographic Breakdown

This heatmap reveals average total sales across customer groups and states. Use it to:

- **Pinpoint high-performing segments and regions.**
- **Identify areas for growth and optimization.**
- **Develop targeted sales and marketing strategies.**
- **Efficiently allocate resources and manage inventory.**
""")