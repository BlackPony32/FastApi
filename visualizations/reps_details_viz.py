import streamlit as st
import pandas as pd
import plotly.express as px
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

def analyze_sales_rep_efficiency(df_pd):
    """Analyzes sales representative efficiency and displays a pie chart."""

    df_pd = df_pd.copy()

    def convert_hours_to_numeric(time_str):
        try:
            hours, minutes = map(int, time_str.split('h '))
            return hours + minutes/60
        except ValueError:
            return pd.NA

    df_pd['Total working hours'] = df_pd['Total working hours'].apply(convert_hours_to_numeric)
    df_pd["Visits per Working Hour"] = df_pd["Total visits"] / df_pd["Total working hours"]
    df_pd["Customers per Visit"] = df_pd["Assigned customers"] / df_pd["Total visits"]

    grouped = df_pd.groupby("Role")[["Total visits"]].sum().reset_index()

    fig = px.pie(
        grouped,
        values='Total visits',
        names='Role',
        title="Distribution of Total Visits by Role",
        color_discrete_sequence=px.colors.qualitative.Set2,  # Updated color palette
        hole=0.3  # Add a hole in the center for a donut chart
    )

    fig.update_traces(textposition='inside', textinfo='percent+label') # Display percentage and label inside
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_font=dict(size=20)  # Increase title font size 
    )

    st.plotly_chart(fig)

    st.markdown("""
    ## Visit Distribution: Understanding Role Contributions

    This interactive donut chart provides a clear picture of how total visits are distributed among your sales roles. By visualizing these proportions, you can gain a better understanding of each role's contribution to overall sales efforts. 
    """)

#Visualizing Customer Engagement: Active Customers vs. Total Visits
def plot_active_customers_vs_visits(df_pd):
    sales_data = df_pd[df_pd["Role"] == "SALES"]

    fig = px.scatter(sales_data, x="Active customers", y="Total visits", color="Name", 
                     trendline="ols", title="Active Customers vs. Total Visits (Sales Reps)")
    fig.update_layout(xaxis_title="Active Customers", yaxis_title="Total Visits")
    st.plotly_chart(fig)
    st.markdown("""
## Customer Engagement vs. Sales Activity

This scatter plot explores the relationship between the number of active customers a sales representative handles and their total number of visits.  By analyzing individual performance and overall trends, you can identify opportunities to optimize sales strategies, resource allocation, and potentially set more effective goals. 
""")
#Travel Distance vs. Number of Visits
def plot_travel_efficiency_line(df_pd):
    """Plots a scatter plot to visualize travel efficiency."""

    df_pd = df_pd.copy()

    # Extract numeric part from "Total travel distance"
    df_pd["Total travel distance"] = df_pd["Total travel distance"].str.extract('(\d+\.?\d*)').astype(float)

    fig = px.scatter(
        df_pd,
        x="Total travel distance",
        y="Total visits",
        color="Role",
        title="Travel Efficiency: Distance vs. Visits",
        trendline="ols",  # Add a trendline
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    fig.update_layout(
        xaxis_title="Total Travel Distance (miles)",
        yaxis_title="Total Visits",
        legend=dict(title="Role", y=0.95, x=0.85)
    )

    st.plotly_chart(fig)

    st.markdown("""
    ## Travel Efficiency: Distance vs. Visits 

    This scatter plot helps you understand the relationship between the distance traveled and the number of visits made by your sales team. Use it to identify trends, potential outliers, and opportunities to optimize travel routes for greater efficiency. 
    """)

#Pure work time per Employee
def analyze_work_hours_and_distance(df_pd):
    """
    Calculates clear work hours and visualizes both clear work hours and
    total travel distance in separate tabs.
    """
    df_pd = df_pd.copy()

    def parse_time(time_str):
        if pd.isna(time_str):
            return 0
        import re
        match = re.match(r'(\d+)h\s*(\d+)m', time_str)
        if match:
            h, m = map(int, match.groups())
        else:
            match = re.match(r'(\d+)m', time_str)
            if match:
                m = int(match.group(1))
                h = 0
            else:
                h, m = 0, 0
        return h + m / 60

    df_pd['Total working hours'] = df_pd['Total working hours'].apply(parse_time)
    df_pd['Total break hours'] = df_pd['Total break hours'].apply(parse_time)
    df_pd['Pure Work Hours'] = df_pd['Total working hours'] - df_pd['Total break hours']
    df_pd = df_pd.sort_values(by='Pure Work Hours', ascending=False).head(10)

    # Extract numeric part from "Total travel distance" 
    df_pd["Total travel distance"] = df_pd["Total travel distance"].str.extract('(\d+\.?\d*)').astype(float)

    tab1, tab2 = st.tabs(["Pure Work Hours", "Total Travel Distance"])

    with tab1:
        st.subheader("Top 10 Employees by Work Hours")
        fig = px.bar(df_pd, x='Name', y='Pure Work Hours',
                     title="Top 10 Employees by Work Hours",
                     text='Pure Work Hours',
                     template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Light24)
        fig.update_traces(texttemplate='%{text:.1f}h', textposition='outside')
        fig.update_layout(xaxis_tickangle=45, yaxis_title="Hours", xaxis_title="Employee")
        st.plotly_chart(fig)

    with tab2:
        st.subheader("Top 10 Employees by Travel Distance")
        fig = px.bar(df_pd, x='Name', y='Total travel distance',
                    title="Top 10 Employees by Travel Distance",
                     text='Total travel distance',
                     template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Light24)
        fig.update_traces(texttemplate='%{text:.1f} mi', textposition='outside')
        fig.update_layout(xaxis_tickangle=45, yaxis_title="Miles", xaxis_title="Employee")
        st.plotly_chart(fig)

    st.markdown("""
    ## Workload and Travel: Insights into Top Performers

    These bar charts, separated into tabs for easy navigation, highlight the top 10 employees based on pure work hours and total travel distance. Use these visualizations to identify potential workload imbalances, analyze travel patterns, and explore ways to optimize efficiency and resource allocation.
    """)

#Total Visits vs. Total Photos Taken
def plot_visits_vs_photos_separate(df_pd):
    roles = df_pd['Role'].unique()

    tabs = st.tabs([role for role in roles]) 

    for i, role in enumerate(roles):
        with tabs[i]:
            st.subheader(f"Visits vs. Photos ({role})")
            role_data = df_pd[df_pd['Role'] == role]
            fig = px.scatter(
                role_data, 
                x="Total visits", 
                y="Total photos", 
                title=f"Visits vs. Photos ({role})",
                template="plotly_white",
                trendline="ols",
                color_discrete_sequence=px.colors.qualitative.Light24  
            )
            fig.update_layout(
                xaxis_title="Total Visits", 
                yaxis_title="Total Photos",
            )
            st.plotly_chart(fig)

    st.markdown("""
    ## Visits vs. Photos: Exploring the Relationship

    These scatter plots analyze the relationship between total visits and the number of photos taken by sales representatives for each role. This visualization can reveal insights into engagement levels, photo-taking patterns, and potential areas for improving data collection or performance. 
    """)

#Exploring Customer Distribution Across Sales Representatives
import streamlit as st
import plotly.express as px
import pandas as pd

def analyze_customer_distribution(df_pd):
    sales_data = df_pd[df_pd["Role"] == "SALES"].copy()

    st.subheader("Distribution of Assigned Customers")
    fig = px.box(
        sales_data, 
        x="Assigned customers", 
        points="all", 
        title="Assigned Customers per Sales Rep",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Light24 
    )
    fig.update_layout(xaxis_title="Number of Assigned Customers")
    st.plotly_chart(fig)

    st.markdown("""
    ## Customer Allocation: Understanding the Distribution

    This box plot reveals the distribution of assigned customers per sales representative, providing insights into potential workload imbalances or variations in customer assignments. Analyzing this distribution can guide decisions related to optimizing sales territories and ensuring a more balanced workload across your team.
    """)
