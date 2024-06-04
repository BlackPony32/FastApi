import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import calendar
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

#Visualize the relationships between Orders/Cases Sold and Revenue
def plot_sales_relationships(df):
    tab1, tab2 = st.tabs(["Orders vs. Revenue", "Cases Sold vs. Revenue"])

    with tab1:
        st.subheader("Orders vs. Revenue")
        fig = px.scatter(df, x="Orders", y="Total revenue", trendline="ols", opacity=0.7,
                         template="plotly_white", color_discrete_sequence=px.colors.qualitative.Light24)
        fig.update_layout(xaxis_title="Orders", yaxis_title="Total Revenue")
        st.plotly_chart(fig)

    with tab2:
        st.subheader("Cases Sold vs. Revenue")
        fig = px.scatter(df, x="Cases sold", y="Total revenue", trendline="ols", opacity=0.7,
                         template="plotly_white", color_discrete_sequence=px.colors.qualitative.Antique)
        fig.update_layout(xaxis_title="Cases Sold", yaxis_title="Total Revenue")
        st.plotly_chart(fig)

    st.markdown("""
    ## Revenue Drivers: Orders and Cases Sold

    These scatter plots analyze the relationships between revenue, orders placed, and cases sold. Explore these visualizations to identify key revenue drivers and understand how order volume and sales volume individually influence your bottom line. This can guide your sales strategies for maximizing revenue growth.
    """)

#Revenue by Month and Role
def plot_revenue_by_month_and_role(df):
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Total revenue'] = df['Total revenue']
    grouped_data = df.groupby(['Month', 'Role'])['Total revenue'].sum().unstack(fill_value=0)

    fig = px.bar(
        grouped_data, 
        x=grouped_data.index, 
        y=grouped_data.columns, 
        title="Revenue by Month and Role", 
        text_auto='.2s',
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Light24
    )
    
    fig.update_layout(xaxis_title="Month", yaxis_title="Total Revenue")
    fig.update_xaxes(tickmode='array', tickvals=grouped_data.index, 
                     ticktext=[calendar.month_name[m] for m in grouped_data.index])
    
    st.plotly_chart(fig)

    st.markdown("""
    ## Revenue Trends: Monthly Performance by Role

    This bar chart presents a breakdown of revenue generated each month, categorized by sales role. Analyze these trends to identify periods of strong performance, potential seasonal variations, and opportunities for targeted improvements in specific months or for particular roles. 
    """)

#Visualize visits and travel distance for each name
def plot_visits_and_travel_distance_by_name(df):
    df['Travel distance'] = pd.to_numeric(df['Travel distance'].str.replace(' mi', ''))
    grouped_data = df.groupby('Name')[['Visits', 'Travel distance']].sum()

    fig = px.bar(
        grouped_data, 
        x=grouped_data.index, 
        y=grouped_data.columns, 
        barmode='group',
        title="Visits and Travel Distance by Rep", 
        text_auto='.2s',
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Light24
    )
    fig.update_layout(xaxis_title="Name", yaxis_title="Count / Distance")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)

    st.markdown("""
    ## Individual Performance: Visits and Travel

    This bar chart provides a comparative view of the total visits and travel distance covered by each sales representative. By analyzing individual performance metrics, you can identify top performers, potential areas for improvement in travel efficiency, and opportunities for optimized resource allocation.
    """)

#Visualize the number of cases sold for each day of the week
def plot_cases_sold_by_day_of_week(df):
    df['Day of Week'] = pd.to_datetime(df['Date']).dt.dayofweek
    weekday_counts = df['Day of Week'].value_counts().sort_index()

    fig = px.line(
        weekday_counts, 
        x=weekday_counts.index, 
        y=weekday_counts.values,
        title="Cases Sold by Day of the Week", 
        markers=True,  # Show markers for data points
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Light24
    )
    
    fig.update_layout(xaxis_title="Day of the Week", yaxis_title="Cases Sold")
    fig.update_xaxes(tickmode='array', tickvals=weekday_counts.index, 
                     ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    st.plotly_chart(fig)

    st.markdown("""
    ## Sales Patterns: Cases Sold by Day of the Week

    This line chart presents the number of cases sold for each day of the week, highlighting the weekly sales trend. By analyzing these patterns, you can identify peak sales days, understand customer behavior, and optimize resource allocation, such as staffing and marketing efforts, to align with weekly sales trends.
    """)

#Visualizing Revenue Trends over Time for Each Role
def plot_revenue_trend_by_month_and_role(df):
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    monthly_revenue = df.groupby(['Month', 'Role'])['Total revenue'].sum().unstack(fill_value=0)

    fig = px.line(
        monthly_revenue, 
        x=monthly_revenue.index, 
        y=monthly_revenue.columns,
        markers=True, 
        title="Revenue Trend by Month and Role",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Light24
    )
    
    fig.update_layout(xaxis_title="Month", yaxis_title="Total Revenue")
    fig.update_xaxes(tickmode='array', tickvals=monthly_revenue.index, 
                     ticktext=[calendar.month_abbr[m] for m in monthly_revenue.index])
    
    st.plotly_chart(fig)

    st.markdown("""
    ## Revenue Trends: Monthly Performance by Role

    This line chart tracks the revenue generated by Merchandisers and Sales Representatives each month, allowing you to visualize revenue fluctuations and compare performance trends between roles. Analyze these trends to identify seasonal patterns, the impact of sales strategies, and opportunities for growth.
    """)
#Exploring the Relationship Between Visits and Orders
def plot_orders_vs_visits_with_regression(df):
    fig = px.scatter(
        df, 
        x="Visits", 
        y="Orders", 
        trendline="ols", 
        title="Visits vs. Orders: Exploring the Relationship",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_layout(xaxis_title="Visits", yaxis_title="Orders")
    st.plotly_chart(fig)

    st.markdown("""
    ## Visits vs. Orders: Exploring the Relationship

    This scatter plot, enhanced with a regression line, visualizes the relationship between the number of visits made by sales representatives and the number of orders generated. Analyze this visualization to understand the correlation between visits and orders, identify potential outliers, and gain insights into the effectiveness of sales efforts.
    """)

#Comparing Performance Metrics for Different Roles
def plot_multiple_metrics_by_role(df):
    grouped_data = df.groupby('Role')[['Visits', 'Orders', 'Cases sold']].sum()

    fig = px.bar(
        grouped_data, 
        x=grouped_data.index, 
        y=grouped_data.columns, 
        barmode='group',
        title="Performance Metrics by Role", 
        text_auto='.2s',
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(xaxis_title="Role", yaxis_title="Count")
    st.plotly_chart(fig)

    st.markdown("""
    ## Comparing Performance: A Role-Based View

    This bar chart provides a comparative overview of key performance metrics (visits, orders, and cases sold) across different sales roles. Analyzing these metrics together can help you identify which roles are excelling in specific areas and pinpoint opportunities for improvement. 
    """)

#Identifying Potential High-Value Clients
def plot_revenue_vs_cases_sold_with_size_and_color(df):
    df = df.copy() # Create a copy of the DataFrame
    if not pd.api.types.is_numeric_dtype(df['Travel distance']):
        df['Travel distance'] = pd.to_numeric(df['Travel distance'].str.replace(' mi', '')) 
    fig = px.scatter(
        df, 
        x="Cases sold", 
        y="Total revenue", 
        size="Visits", 
        color="Travel distance",
        title="Revenue vs. Cases Sold: Insights from Visits and Travel", 
        hover_data=df.columns, 
        opacity=0.7,
        template="plotly_white",
        color_continuous_scale=px.colors.sequential.Viridis  
    )
    
    fig.update_layout(xaxis_title="Cases Sold", yaxis_title="Total Revenue")
    fig.update_traces(marker=dict(sizemode='area', sizeref=2.*max(df['Visits'])/(40.**2))) 
    st.plotly_chart(fig)

    st.markdown("""
    ## Multifaceted Sales Analysis: Revenue, Cases Sold, Visits, and Travel 

    This interactive scatter plot provides a comprehensive view of your sales data. Explore the relationship between revenue and cases sold, with the size of each point representing visit frequency and color indicating travel distance. This visualization allows you to uncover deeper insights into the factors influencing sales performance and identify potential areas for optimization.  
    """)
