import streamlit as st
import pandas as pd
import os
import openai
import httpx
import asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.agent_types import AgentType

from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe, Agent

from visualizations import (
    third_party_sales_viz, order_sales_summary_viz, best_sellers_viz,
    reps_details_viz, reps_summary_viz, skus_not_ordered_viz,
    low_stock_inventory_viz, current_inventory_viz, top_customers_viz, customer_details_viz
)
from side_func import identify_file, get_file_name, get_csv_columns
#from main import file_name
load_dotenv()
st.set_page_config(layout="wide")

CHARTS_PATH = "exports/charts/"
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(CHARTS_PATH):
    os.makedirs(CHARTS_PATH)

file_name = get_file_name()
last_uploaded_file_path = os.path.join(UPLOAD_DIR, file_name)

def directory_contains_png_files(directory_path):
    files = os.listdir(directory_path)
    for file in files:
        if file.endswith(".png"):
            return True
    return False

async def read_csv(file_path):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, pd.read_csv, file_path)

async def build_some_chart(df, prompt):
    llm = OpenAI()
    # pandas_ai = SmartDataframe(df, config={"llm": llm})
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    result = pandas_ai.chat(prompt)
    return await result

async def chat_with_file(prompt, file_path):
    file_name = get_file_name()
    last_uploaded_file_path = os.path.join(UPLOAD_DIR, file_name)
    try:
        if last_uploaded_file_path is None or not os.path.exists(last_uploaded_file_path):
            raise HTTPException(status_code=400, detail=f"No file has been uploaded or downloaded yet {last_uploaded_file_path}")
            
        result = chat_with_agent(prompt, last_uploaded_file_path)
        
        return {"response": result}

    except ValueError as e:
        return {"error": f"ValueError: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}

def chat_with_agent(input_string, file_path):
    try:
        # Assuming file_path is always CSV after conversion
        df = pd.read_csv(file_path)
        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
            file_path,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )
        result = agent.invoke(input_string)
        return result['output']
    except ImportError as e:
        raise ValueError("Missing optional dependency 'tabulate'. Use pip or conda to install tabulate.")
    except pd.errors.ParserError as e:
        raise ValueError("Parsing error occurred: " + str(e))
    except Exception as e:
        raise ValueError(f"An error occurred: {str(e)}")


async def main_viz():
    st.title("Report Analysis")

    if os.path.exists(last_uploaded_file_path):
        df = pd.read_csv(last_uploaded_file_path)
        col1, col2 = st.columns([1, 1])
        file_type = identify_file(df)
        
        if file_type == 'Unknown':
            st.warning(f"This is  {file_type} type report,so this is generated report to it")
        else:
            st.success(f"This is  {file_type} type. File is available for visualization.")
        
        with col1:
            st.dataframe(df, use_container_width=True)

        with col2:
            st.info("Chat Below")
            input_text = st.text_area("Enter your query")
            if input_text is not None:
                if st.button("Chat with CSV"):
                    result = await chat_with_file(input_text, last_uploaded_file_path)
                    if "response" in result:
                        st.success(result["response"])
                    else:
                        st.error(result.get("error", "Unknown error occurred"))
            input_text2 = st.text_area("Enter your query for the plot")
            if input_text2 is not None:
                if st.button("Build some chart"):
                    st.info("Plotting your Query: " + input_text2)
                    #result = build_some_chart(df, input_text2)
                    result = test_plot_maker(df, input_text2)
                    #st.success(result)

        # directory_path = "exports/charts/"
        if directory_contains_png_files(CHARTS_PATH):
            with col1:
                st.subheader("Generated Visualisation")
                st.image(f'{CHARTS_PATH}temp_chart.png')
        
        if df.empty:
            st.warning("### This data report is empty - try downloading another one to get better visualizations")
        
        elif file_type == "3rd Party Sales Summary report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                columns = get_csv_columns(last_uploaded_file_path)
                third_party_sales_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))

                if "Product name" in columns and "Grand total" in columns:
                    third_party_sales_viz.visualize_product_analysis(df)
                else:
                    st.warning("There is no Grand total or Product name, so visualizing can not be ready")
                
                if "Customer" in columns and "Product name" in columns and "QTY" in columns and "Grand total" in columns:
                    third_party_sales_viz.visualize_sales_trends(df)
                else:
                    st.warning("There is no Customer or Product name or Quantity or Grand total, so visualizing can not be ready")
                
                if "Product name" in columns and "QTY" in columns and "Grand total" in columns:
                    third_party_sales_viz.visualize_combined_analysis(df)
                else:
                    st.warning("There is no Delivery status or Product name or Quantity or Grand total, so visualizing can not be ready")
                
            with cc2:
                columns = get_csv_columns(last_uploaded_file_path)
                if "Discount type" in columns and "Total invoice discount" in columns:
                    third_party_sales_viz.visualize_discount_analysis(df)
                else:
                    st.warning("There is no Discount type or Total invoice discount, so visualizing can not be ready")
                # line_chart_plotly()
                if "Discount type" in columns:
                    third_party_sales_viz.analyze_discounts(df)
                else:
                    st.warning("There is no Discount type, so visualizing can not be ready")
                
                if "Grand total" in columns and "Manufacturer specific discount" in columns and "Customer discount" in columns:
                    third_party_sales_viz.area_visualisation(df)
                else:
                    st.warning("There is no Grand total or Manufacturer specific discount or Customer discount, so visualizing can not be ready")
        elif file_type == "Order Sales Summary report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                columns = get_csv_columns(last_uploaded_file_path)
                #df = order_sales_summary_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                
                if "Customer" in columns and "Product name" in columns and "Created at" in columns:
                    order_sales_summary_viz.visualize_sales_trends(df)
                else:
                    st.warning("There is no Customer or Product name or Created at columns, so visualizing can not be ready")
                
                if "Product name" in columns and "Grand total" in columns:
                    order_sales_summary_viz.visualize_product_analysis(df)
                else:
                    st.warning("There is no Product name or Grand total columns, so visualizing can not be ready")
                
                if "Discount type" in columns and "Total invoice discount" in columns and "Customer" in columns:
                    order_sales_summary_viz.visualize_discount_analysis(df)
                else:
                    st.warning("There is no Discount type or Total invoice discount or Customer columns, so visualizing can not be ready")
                
            with cc2:
                # bar_chart()
                if "Delivery status" in columns and "Delivery methods" in columns:
                    order_sales_summary_viz.visualize_delivery_analysis(df)
                else:
                    st.warning("There is no Delivery status or Delivery methods columns, so visualizing can not be ready")
                
                if "Payment status" in columns:
                    order_sales_summary_viz.visualize_payment_analysis(df)
                else:
                    st.warning("There is no Payment status column, so visualizing can not be ready")
                
                if "Product name" in columns and "Grand total" in columns and "QTY" in columns and "Delivery status"in columns:
                    order_sales_summary_viz.visualize_combined_analysis(df)
                else:
                    st.warning("There is no Grand total or Product name or Quantity or Delivery status columns, so visualizing can not be ready")
                
                # line_chart_plotly()
            # todo check map data  (addresses or coordinates)
            #map_features()
            #pycdeck_map()
        elif file_type == "Best Sellers report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                columns = get_csv_columns(last_uploaded_file_path)
                df = best_sellers_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                
                if "Available cases (QTY)" in columns and "Product name" in columns:
                    best_sellers_viz.create_available_cases_plot(df)
                else:
                    st.warning("There is no Product name or Available cases (QTY) columns, so visualizing can not be ready")
                
                if "Product name" in columns and "Total revenue" in columns and "Cases sold" in columns:
                    best_sellers_viz.product_analysis_app(df)
                else:
                    st.warning("There is no Total revenue or Product name or Cases sold columns, so visualizing can not be ready")
                
                if "Cases sold" in columns and "Total revenue" in columns:
                    best_sellers_viz.create_cases_revenue_relationship_plot(df)
                else:
                    st.warning("There is no Total revenue or Cases sold columns, so visualizing can not be ready")
                
            with cc2:
                # bar_chart()
                if "Category name" in columns and "Wholesale price" in columns and "Retail price" in columns:
                    best_sellers_viz.price_comparison_app(df)
                else:
                    st.warning("There is no Category name or Wholesale price or Retail price columns, so visualizing can not be ready")
                
                if "Total revenue" in columns and "Product name" in columns:
                    best_sellers_viz.create_revenue_vs_profit_plot(df)
                else:
                    st.warning("There is no Total revenue or Product name columns, so visualizing can not be ready")
        elif file_type == "Representative Details report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                columns = get_csv_columns(last_uploaded_file_path)
                df = reps_details_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                
                if "Total working hours" in columns and "Total visits" in columns and "Assigned customers" in columns and "Role"in columns:
                    reps_details_viz.analyze_sales_rep_efficiency(df)
                else:
                    st.warning("There is no Total working hours or Total visits or Assigned customers or Role columns, so visualizing can not be ready")
                
                if "Role" in columns and "Active customers" in columns and "Total visits" in columns:
                    reps_details_viz.plot_active_customers_vs_visits(df)
                else:
                    st.warning("There is no Role or Active customers or Total visits columns, so visualizing can not be ready")
                
                if "Total travel distance" in columns and "Total visits" in columns and "Role" in columns:
                    reps_details_viz.plot_travel_efficiency_line(df)
                else:
                    st.warning("There is no Total travel distance or Total visits or Role columns, so visualizing can not be ready")
                
            with cc2:
                if "Total working hours" in columns and "Total break hours" in columns and "Total travel distance" in columns:
                    reps_details_viz.analyze_work_hours_and_distance(df)
                else:
                    st.warning("There is no Total working hours or Total break hours or Total travel distance columns, so visualizing can not be ready")
                
                if "Role" in columns and "Total visits" in columns and "Total photos" in columns:
                    reps_details_viz.plot_visits_vs_photos_separate(df)
                else:
                    st.warning("There is no Role or Total visits or Total photos columns, so visualizing can not be ready")
                
                if "Role" in columns and "Assigned customers" in columns:
                    reps_details_viz.analyze_customer_distribution(df)
                else:
                    st.warning("There is no Role or Assigned customers columns, so visualizing can not be ready")
        elif file_type == "Reps Summary report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                columns = get_csv_columns(last_uploaded_file_path)
                df = reps_summary_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                
                if "Orders" in columns and "Total revenue" in columns and "Cases sold" in columns:
                    reps_summary_viz.plot_sales_relationships(df)
                else:
                    st.warning("There is no Orders or Total revenue or Cases sold columns, so visualizing can not be ready")
                
                if "Date" in columns and "Role" in columns and "Total revenue" in columns:
                    reps_summary_viz.plot_revenue_by_month_and_role(df)
                else:
                    st.warning("There is no Date or Role or Total revenue columns, so visualizing can not be ready")
                
                if "Date" in columns:
                    reps_summary_viz.plot_cases_sold_by_day_of_week(df)
                else:
                    st.warning("There is no Date column, so visualizing can not be ready")
                
                if "Date" in columns and "Total revenue" in columns and "Role" in columns:
                    reps_summary_viz.plot_revenue_trend_by_month_and_role(df)
                else:
                    st.warning("There is no Date or Total revenue or Role columns, so visualizing can not be ready")
                
            with cc2:
                if "Name" in columns and "Visits" in columns and "Travel distance" in columns:
                    reps_summary_viz.plot_visits_and_travel_distance_by_name(df)
                else:
                    st.warning("There is no Name or Visits or Travel distance columns, so visualizing can not be ready")
                
                if "Visits" in columns and "Orders" in columns:
                    reps_summary_viz.plot_orders_vs_visits_with_regression(df)
                else:
                    st.warning("There is no Visits or Orders columns, so visualizing can not be ready")
                
                if "Role" in columns and "Visits" in columns and "Orders" in columns and "Cases sold" in columns:
                    reps_summary_viz.plot_multiple_metrics_by_role(df)
                else:
                    st.warning("There is no Role or Visits or Orders or Cases sold columns, so visualizing can not be ready")
                
                if "Cases sold" in columns and "Total revenue" in columns and "Visits" in columns and "Travel distance" in columns:
                    reps_summary_viz.plot_revenue_vs_cases_sold_with_size_and_color(df)
                else:
                    st.warning("There is no Cases sold or Total revenue or Visits or Travel distance columns, so visualizing can not be ready")
        elif file_type == "SKU's Not Ordered report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                columns = get_csv_columns(last_uploaded_file_path)
                df = skus_not_ordered_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                
                if "Category name" in columns:
                    skus_not_ordered_viz.create_unordered_products_by_category_plot(df)
                else:
                    st.warning("There is no Category name column, so visualizing can not be ready")
                
                if "Available cases (QTY)" in columns:
                    skus_not_ordered_viz.create_available_cases_distribution_plot(df)
                else:
                    st.warning("There is no Available cases (QTY) column, so visualizing can not be ready")
                
                if "Category name" in columns and "Retail price" in columns and "Available cases (QTY)" in columns:
                    skus_not_ordered_viz.price_vs_available_cases_app(df)
                else:
                    st.warning("There is no Category name or Retail price or Available cases (QTY) columns, so visualizing can not be ready")
                
            with cc2:
                if "Available cases (QTY)" in columns and "Retail price" in columns and "Wholesale price" in columns and "Category name" in columns:
                    skus_not_ordered_viz.create_wholesale_vs_retail_price_scatter(df)
                else:
                    st.warning("There is no Available cases (QTY) or Retail price or Wholesale price columns, so visualizing can not be ready")
                
                if "Category name" in columns and "Retail price" in columns:
                    skus_not_ordered_viz.df_unordered_products_per_category_and_price_range(df)
                else:
                    st.warning("There is no Category name or Retail price columns, so visualizing can not be ready")
        elif file_type == "Low Stock Inventory report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                columns = get_csv_columns(last_uploaded_file_path)
                df = low_stock_inventory_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                
                if "Category name" in columns and "Product name" in columns and "Available cases (QTY)" in columns and "Wholesale price" in columns:
                    low_stock_inventory_viz.low_stock_analysis_app(df)
                else:
                    st.warning("There is no Available cases (QTY) or Category name or Product name columns, so visualizing can not be ready")
                
                if "Retail price" in columns and "Wholesale price" in columns and "Product name" in columns:
                    low_stock_inventory_viz.create_profit_margin_analysis_plot(df)
                else:
                    st.warning("There is no Product name or Retail price or Wholesale price columns, so visualizing can not be ready")
                
                if "Manufacturer name" in columns and "Product name" in columns:
                    low_stock_inventory_viz.create_low_stock_by_manufacturer_bar_plot(df)
                else:
                    st.warning("There is no Manufacturer name or Product name columns, so visualizing can not be ready")
                
            with cc2:
                if "Wholesale price" in columns and "Available cases (QTY)" in columns:
                    low_stock_inventory_viz.create_interactive_price_vs_quantity_plot(df)
                else:
                    st.warning("There is no Available cases (QTY) or Wholesale price columns, so visualizing can not be ready")
                
                if "Retail price" in columns and "Available cases (QTY)" in columns and "Product name" in columns:
                    low_stock_inventory_viz.create_quantity_price_ratio_plot(df)
                else:
                    st.warning("There is no Available cases (QTY) or Retail price columns, so visualizing can not be ready")
        elif file_type == "Current Inventory report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                columns = get_csv_columns(last_uploaded_file_path)
                df = current_inventory_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                
                if "Available cases (QTY)" in columns and "Wholesale price" in columns and "Category name" in columns:
                    current_inventory_viz.df_analyze_inventory_value_by_category(df)
                else:
                    st.warning("There is no Available cases (QTY) or Wholesale price columns, so visualizing can not be ready")
                
                if "Available cases (QTY)" in columns and "Retail price" in columns and "Category name" in columns and "Wholesale price" in columns:
                    current_inventory_viz.df_analyze_quantity_vs_retail_price(df)
                else:
                    st.warning("There is no Available cases (QTY) or Retail price or Category name columns, so visualizing can not be ready")
                
                if "Available cases (QTY)" in columns and "Wholesale price" in columns and "Manufacturer name" in columns:
                    current_inventory_viz.df_analyze_inventory_value_by_manufacturer(df)
                else:
                    st.warning("There is no Available cases (QTY) or Manufacturer name or Wholesale price columns, so visualizing can not be ready")
                
            with cc2:
                if "Wholesale price" in columns and "Available cases (QTY)" in columns and "Product name" in columns:
                    current_inventory_viz.df_analyze_inventory_value_per_unit(df)
                else:
                    st.warning("There is no Product name or Available cases (QTY) or Wholesale price columns, so visualizing can not be ready")
                
                if "Retail price" in columns and "Category name" in columns:
                    current_inventory_viz.df_compare_average_retail_prices(df)
                else:
                    st.warning("There is no Category name or Retail price columns, so visualizing can not be ready")
        elif file_type == "Top Customers report":
            cc1, cc2 = st.columns([1,1])
            with cc1:
                columns = get_csv_columns(last_uploaded_file_path)
                df = top_customers_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                if "Name" in columns and "Total sales" in columns and "Territory" in columns and "Payment terms" in columns:
                    top_customers_viz.customer_analysis_app(df)
                else:
                    st.warning("There is no Name or Total sales or Territory or Payment terms columns, so visualizing can not be ready")
                
                if "Payment terms" in columns:
                    top_customers_viz.interactive_bar_plot_app(df)
                else:
                    st.warning("There is no Payment terms column, so visualizing can not be ready")
                
            with cc2:
                if "Total sales" in columns:
                    top_customers_viz.create_non_zero_sales_grouped_plot(df)
                else:
                    st.warning("There is no Total sales column, so visualizing can not be ready")
                
                if "Group" in columns and "Billing city" in columns:
                    top_customers_viz.interactive_group_distribution_app(df)
                else:
                    st.warning("There is no Group or Billing city columns, so visualizing can not be ready")
        elif file_type == "Customer Details report":
            cc1, cc2 = st.columns([1,1])
            with cc1:
                columns = get_csv_columns(last_uploaded_file_path)
                df = customer_details_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                
                if "Group" in columns and "Total orders" in columns and "Total sales" in columns:
                    customer_details_viz.plot_orders_and_sales_plotly(df)
                else:
                    st.warning("There is no Group or Total orders or Total sales columns, so visualizing can not be ready")
                
                if "Payment terms" in columns:
                    customer_details_viz.bar_plot_sorted_with_percentages(df)
                else:
                    st.warning("There is no Payment terms column, so visualizing can not be ready")
                
            with cc2:
                if "Total sales" in columns:
                    customer_details_viz.create_interactive_non_zero_sales_plot(df)
                else:
                    st.warning("There is no Total sales column, so visualizing can not be ready")
                
                    
                if "Total sales" in columns and "Group" in columns and "Billing state" in columns:
                    customer_details_viz.create_interactive_average_sales_heatmap(df)
                else:
                    st.warning("There is no Total sales or Group or Billing state columns, so visualizing can not be ready")


        else:
            df = customer_details_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
            #here can turn on lida and try to analyze dataset automatically by its toolset
            #lida_call(query=input_text, df=df)
            st.write(big_summary(last_uploaded_file_path))
            summary_lida(df)
            

    else:
        st.warning("No file has been uploaded or downloaded yet.")


def big_summary(file_path):
    try:
        prompt = f"""
        I have a CSV file that contains important business data.
        I need a comprehensive and easy-to-read summary of this data that would be useful for a business owner.
        The summary should include key insights, trends, and any significant patterns or anomalies found in the data.
        Please ensure the summary is concise and written in layman's terms, focusing on actionable insights
        that can help in decision-making.
        """
        result = chat_with_agent(prompt, file_path)
        
        return result

    except ValueError as e:
        return {"error": f"ValueError: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}

def test_plot_maker(df, text):
    from lida import Manager, TextGenerationConfig, llm
    from lida.datamodel import Goal
    lida = Manager(text_gen = llm("openai")) 

    visualization_libraries = "plotly"
    i = 0

    goals = [text]

    textgen_config = TextGenerationConfig(n=1, 
                                      temperature=0.1, model="gpt-4o", 
                                      use_cache=True)

    summary = lida.summarize(df, 
                summary_method="default", textgen_config=textgen_config) 
    textgen_config = TextGenerationConfig(n=1, temperature=0.1, model="gpt-4o", use_cache=True)
    visualizations = lida.visualize(summary=summary, goal=goals[0], textgen_config=textgen_config, library=visualization_libraries)
    if visualizations:  # Check if the visualizations list is not empty
        selected_viz = visualizations[0]
        exec_globals = {'data': df}
        exec(selected_viz.code, exec_globals)
        st.plotly_chart(exec_globals['chart'])
    else:
        st.warning("No visualizations were generated for this query.")    




def summary_lida(df):
    from lida import Manager, TextGenerationConfig, llm
    from lida.datamodel import Goal
    lida = Manager(text_gen = llm("openai")) 
    textgen_config = TextGenerationConfig(n=1, 
                                      temperature=0.1, model="gpt-3.5-turbo-0301", 
                                      use_cache=True)
    # load csv datset
    summary = lida.summarize(df, 
                summary_method="default", textgen_config=textgen_config)     

    goals = lida.goals(summary, n=6, textgen_config=textgen_config)
    visualization_libraries = "plotly"

    cc1, cc2 = st.columns([1,1])
    num_visualizations = 2

    i = 0
    for i, goal in enumerate(goals):
        if i < 3:
            with cc1:
                st.write("The question for the report was generated by artificial intelligence: " + goals[i].question)
                textgen_config = TextGenerationConfig(n=num_visualizations, temperature=0.1, model="gpt-3.5-turbo-0301", use_cache=True)
                visualizations = lida.visualize(summary=summary,goal=goals[i],textgen_config=textgen_config,library=visualization_libraries)
                if visualizations:  # Check if the visualizations list is not empty
                    selected_viz = visualizations[0]
                    exec_globals = {'data': df}
                    exec(selected_viz.code, exec_globals)
                    st.plotly_chart(exec_globals['chart'])
                else:
                    st.write("No visualizations were generated for this goal.")
                
                st.write("### Explanation of why this question can be useful: " + goals[i].rationale)
                st.write("Method of visualization: " + goals[i].visualization)
        else:
            with cc2:
                st.write("The question for the report was generated by artificial intelligence: " + goals[i].question)
                
                textgen_config = TextGenerationConfig(n=num_visualizations, temperature=0.1, model="gpt-3.5-turbo-0301", use_cache=True)
                visualizations = lida.visualize(summary=summary,goal=goals[i],textgen_config=textgen_config,library=visualization_libraries)
                
                if visualizations:  # Check if the visualizations list is not empty
                    selected_viz = visualizations[0]
                    exec_globals = {'data': df}
                    exec(selected_viz.code, exec_globals)
                    st.plotly_chart(exec_globals['chart'])
                else:
                    st.write("No visualizations were generated for this goal.")
                
                st.write("### Explanation of why this question can be useful: " + goals[i].rationale)
                st.write("Method of visualization: " + goals[i].visualization)


if __name__ == "__main__":
    asyncio.run(main_viz())
