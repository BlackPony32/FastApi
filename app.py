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
from side_func import identify_file, get_file_name
#from main import file_name
load_dotenv()
st.set_page_config(layout="wide")

#openai_api_key = os.getenv("OPENAI_API_KEY")

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
            st.info("Chat Below 2 2 2 2")
            input_text2 = st.text_area("Enter your query for the plot")
            if input_text2 is not None:
                if st.button("Build some chart"):
                    st.info("Your Query: " + input_text2)
                    result = build_some_chart(df, input_text2)
                    st.success(result)

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
                third_party_sales_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                third_party_sales_viz.visualize_product_analysis(df)
                third_party_sales_viz.visualize_sales_trends(df)
                third_party_sales_viz.visualize_combined_analysis(df)

            with cc2:
                # bar_chart()
                third_party_sales_viz.visualize_discount_analysis(df)
                # line_chart_plotly()
                third_party_sales_viz.analyze_discounts(df)
                third_party_sales_viz.area_visualisation(df)
        elif file_type == "Order Sales Summary report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                #df = order_sales_summary_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                order_sales_summary_viz.visualize_sales_trends(df)
                order_sales_summary_viz.visualize_product_analysis(df)
                order_sales_summary_viz.visualize_discount_analysis(df)
            with cc2:
                # bar_chart()
                order_sales_summary_viz.visualize_delivery_analysis(df)
                order_sales_summary_viz.visualize_payment_analysis(df)
                order_sales_summary_viz.visualize_combined_analysis(df)
                # line_chart_plotly()
            # todo check map data  (addresses or coordinates)
            #map_features()
            #pycdeck_map()
        elif file_type == "Best Sellers report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                df = best_sellers_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                best_sellers_viz.create_available_cases_plot(df)
                best_sellers_viz.product_analysis_app(df)
                best_sellers_viz.create_cases_revenue_relationship_plot(df)
            with cc2:
                # bar_chart()
                best_sellers_viz.price_comparison_app(df)
                best_sellers_viz.create_revenue_vs_profit_plot(df)
        elif file_type == "Representative Details report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                df = reps_details_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                reps_details_viz.analyze_sales_rep_efficiency(df)
                reps_details_viz.plot_active_customers_vs_visits(df)
                reps_details_viz.plot_travel_efficiency_line(df)
            with cc2:
                reps_details_viz.analyze_work_hours_and_distance(df)
                reps_details_viz.plot_visits_vs_photos_separate(df)
                reps_details_viz.analyze_customer_distribution(df)
        elif file_type == "Reps Summary report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                df = reps_summary_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                reps_summary_viz.plot_sales_relationships(df)
                reps_summary_viz.plot_visits_and_travel_distance_by_name(df)
                reps_summary_viz.plot_cases_sold_by_day_of_week(df)
                reps_summary_viz.plot_revenue_trend_by_month_and_role(df)
            with cc2:
                reps_summary_viz.plot_revenue_by_month_and_role(df)
                reps_summary_viz.plot_orders_vs_visits_with_regression(df)
                reps_summary_viz.plot_multiple_metrics_by_role(df)
                reps_summary_viz.plot_revenue_vs_cases_sold_with_size_and_color(df)
        elif file_type == "SKU's Not Ordered report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                df = skus_not_ordered_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                skus_not_ordered_viz.create_unordered_products_by_category_plot(df)
                skus_not_ordered_viz.create_available_cases_distribution_plot(df)
                skus_not_ordered_viz.price_vs_available_cases_app(df)
            with cc2:
                skus_not_ordered_viz.create_wholesale_vs_retail_price_scatter(df)
                skus_not_ordered_viz.df_unordered_products_per_category_and_price_range(df)
        elif file_type == "Low Stock Inventory report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                df = low_stock_inventory_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                low_stock_inventory_viz.low_stock_analysis_app(df)
                low_stock_inventory_viz.create_profit_margin_analysis_plot(df)
                low_stock_inventory_viz.create_low_stock_by_manufacturer_bar_plot(df)
            with cc2:
                low_stock_inventory_viz.create_interactive_price_vs_quantity_plot(df)
                low_stock_inventory_viz.create_quantity_price_ratio_plot(df)
        elif file_type == "Current Inventory report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                df = current_inventory_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                current_inventory_viz.df_analyze_inventory_value_by_category(df)
                current_inventory_viz.df_analyze_quantity_vs_retail_price(df)
                current_inventory_viz.df_analyze_inventory_value_by_manufacturer(df)
            with cc2:
                current_inventory_viz.df_analyze_inventory_value_per_unit(df)
                current_inventory_viz.df_compare_average_retail_prices(df)
        elif file_type == "Top Customers report":
            cc1, cc2 = st.columns([1,1])
            with cc1:
                df = top_customers_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                top_customers_viz.customer_analysis_app(df)
                top_customers_viz.interactive_bar_plot_app(df)

            with cc2:
                top_customers_viz.create_non_zero_sales_grouped_plot(df)
                top_customers_viz.interactive_group_distribution_app(df)
        elif file_type == "Customer Details report":
            cc1, cc2 = st.columns([1,1])
            with cc1:
                df = customer_details_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
                customer_details_viz.plot_orders_and_sales_plotly(df)
                customer_details_viz.bar_plot_sorted_with_percentages(df)
            with cc2:
                customer_details_viz.create_interactive_non_zero_sales_plot(df)
                customer_details_viz.create_interactive_average_sales_heatmap(df)

        else:
            df = customer_details_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
            #here can turn on lida and try to analyze dataset automatically by its toolset
            #lida_call(query=input_text, df=df)
            st.write(big_summary(last_uploaded_file_path))
            test_lida(df)
            

    else:
        st.warning("No file has been uploaded or downloaded yet.")

def lida_call(query, df):
    from lida import Manager, TextGenerationConfig, llm
    from lida.datamodel import Goal


    prompt_content = f"""
    The dataset is ALREADY loaded into a DataFrame named 'df'. DO NOT load the data again.
    always start code with import useful libs like pyplot
    Try to make the graphs more attractive and if you can add columns label to it
    Before plotting, ensure the data is ready use this code:
    data['Created at'] = pd.to_datetime(data['Created at'])
    # Identify numeric columns automatically
    numeric_cols = data.select_dtypes(include=np.number).columns
    data[numeric_cols] = data[numeric_cols].replace('[$,]', '', regex=True).astype(float)
    Use package Pandas and Matplotlib ONLY.
    Provide SINGLE CODE BLOCK with a solution using Pandas and Matplotlib plots in a single figure to address the following query:
    {query}
    - USE SINGLE CODE BLOCK with a solution. 
    - Do NOT EXPLAIN the code 
    - DO NOT COMMENT the code. 
    - ALWAYS WRAP UP THE CODE IN A SINGLE CODE BLOCK.
    - The code block must start and end with ```
        
    - Example code format ```code```
    
    - Colors to use for background and axes of the figure : #F0F0F6
    - Try to use the following color palette for coloring the plots : #8f63ee #ced5ce #a27bf6 #3d3b41
        
    """
    # Define the messages for the OpenAI model
    messages = [
        {
            "role": "system",
            "content": "You are a helpful Data Visualization assistant who gives a single block without explaining or commenting the code to plot. IF ANYTHING NOT ABOUT THE DATA, JUST politely respond that you don't know.",
        },
        {"role": "user", "content": prompt_content},
    ]
    # Call OpenAI and display the response
    with st.status("📟 *Prompting is the new programming*..."):
        with st.chat_message("assistant", avatar="📊"):
            botmsg = st.empty()
            response = []
            for chunk in openai.ChatCompletion.create(
                model='gpt-3.5-turbo', messages=messages, stream=True
            ):
                text = chunk.choices[0].get("delta", {}).get("content")
                if text:
                    response.append(text)
                    result = "".join(response).strip()
                    botmsg.write(result)
    execute_openai_code(result, df, query)

    if last_uploaded_file_path is not None:
        data = pd.read_csv('uploads/last_uploaded.csv')
        selected_dataset = data

    # summarization_methods = ["default", "llm", "columns"]
    summarization_methods = [
        {"label": "llm",
         "description":
         "Uses the LLM to generate annotate the default summary, adding details such as semantic types for columns and dataset description"},
]

    # selected_method = st.sidebar.selectbox("Choose a method", options=summarization_methods)
    selected_method_label = st.sidebar.selectbox(
        'Choose a method',
        options=[method["label"] for method in summarization_methods],
        index=0
    )

    selected_method = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["label"]

    # add description of selected method in very small font to sidebar
    selected_summary_method_description = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["description"]

    if selected_method:
        st.sidebar.markdown(
            f"<span> {selected_summary_method_description} </span>",
            unsafe_allow_html=True)
        

    # Step 3 - Generate data summary
    if selected_method:
        lida = Manager(text_gen=llm("openai", api_key=openai_key))
        textgen_config = TextGenerationConfig(
            n=1,
            temperature=temperature,
            model=selected_model,
            use_cache=use_cache)

        st.write("## Summary")
        # **** lida.summarize *****
        summary = lida.summarize(
            selected_dataset,
            summary_method=selected_method,
            textgen_config=textgen_config)

        if "dataset_description" in summary:
            st.write(summary["dataset_description"])

        if "fields" in summary:
            fields = summary["fields"]
            nfields = []
            for field in fields:
                flatted_fields = {}
                flatted_fields["column"] = field["column"]
                # flatted_fields["dtype"] = field["dtype"]
                for row in field["properties"].keys():
                    if row != "samples":
                        flatted_fields[row] = field["properties"][row]
                    else:
                        flatted_fields[row] = str(field["properties"][row])
                # flatted_fields = {**flatted_fields, **field["properties"]}
                nfields.append(flatted_fields)
            nfields_df = pd.DataFrame(nfields)
            st.write(nfields_df)
        else:
            st.write(str(summary))

        # Step 4 - Generate goals
        if summary:
            # **** lida.goals *****
            goals = lida.goals(summary, n=5, textgen_config=textgen_config)
            st.write(f"## Goals ({len(goals)})")

            prompt_content = f"""
                The dataset is ALREADY loaded into a DataFrame named 'df'. DO NOT load the data again.
                always start code with import useful libs like pyplot
                Try to make the graphs more attractive and if you can add columns label to it

                Before plotting, ensure the data is ready use this code:
                data['Created at'] = pd.to_datetime(data['Created at'])

                # Identify numeric columns automatically
                numeric_cols = data.select_dtypes(include=np.number).columns
                data[numeric_cols] = data[numeric_cols].replace('[$,]', '', regex=True).astype(float)

                Use package Pandas and Matplotlib ONLY.
                Provide SINGLE CODE BLOCK with a solution using Pandas and Matplotlib plots in a single figure to address the following query:

                {goals[0].question}

                - USE SINGLE CODE BLOCK with a solution.
                - Do NOT EXPLAIN the code
                - DO NOT COMMENT the code.
                - ALWAYS WRAP UP THE CODE IN A SINGLE CODE BLOCK.
                - The code block must start and end with ```

                - Example code format ```code```

                - Colors to use for background and axes of the figure : #F0F0F6
                - Try to use the following color palette for coloring the plots : #8f63ee #ced5ce #a27bf6 #3d3b41

                """

            default_goal = goals[0].question
            goal_questions = [goal.question for goal in goals]

            selected_goal = st.selectbox('Choose a generated goal', options=goal_questions, index=0)

            # st.markdown("### Selected Goal")
            selected_goal_index = goal_questions.index(selected_goal)
            st.write(goals[selected_goal_index])

            selected_goal_object = goals[selected_goal_index]

            # Step 5 - Generate visualizations
            if selected_goal_object:
                visualization_libraries = 'seaborn'

                selected_library = st.sidebar.selectbox(
                    'Choose a visualization library',
                    options=visualization_libraries,
                    index=0
                )

                # Update the visualization generation call to use the selected library.
                st.write("## Visualizations")

                # slider for number of visualizations
                num_visualizations = st.sidebar.slider(
                    "Number of visualizations to generate",
                    min_value=1,
                    max_value=10,
                    value=2)

                textgen_config = TextGenerationConfig(
                    n=num_visualizations, temperature=temperature,
                    model=selected_model,
                    use_cache=use_cache)

                # **** lida.visualize *****
                visualizations = lida.visualize(
                    summary=summary,
                    goal=selected_goal_object,
                    textgen_config=textgen_config,
                    library=selected_library)

                viz_titles = [f'Visualization {i+1}' for i in range(len(visualizations))]

                selected_viz_title = st.selectbox('Choose a visualization', options=viz_titles, index=0)

                selected_viz = visualizations[viz_titles.index(selected_viz_title)]

                if selected_viz.raster:
                    from PIL import Image
                    import io
                    import base64

                    imgdata = base64.b64decode(selected_viz.raster)
                    img = Image.open(io.BytesIO(imgdata))
                    st.image(img, caption=selected_viz_title, use_column_width=True)

                st.write("### Visualization Code")
                st.code(selected_viz.code)

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


def test_lida(df):
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
