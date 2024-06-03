import streamlit as st
import pandas as pd
import os
import asyncio
from visualizations import (
    third_party_sales_viz, order_sales_summary_viz, best_sellers_viz,
    reps_details_viz, reps_summary_viz, skus_not_ordered_viz,
    low_stock_inventory_viz, current_inventory_viz, top_customers_viz, customer_details_viz
)
from side_func import identify_file, get_file_name
#from main import file_name

st.set_page_config(layout="wide")

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


file_name = get_file_name()
last_uploaded_file_path = os.path.join(UPLOAD_DIR, file_name)


async def read_csv(file_path):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, pd.read_csv, file_path)

async def chat_with_csv(df, prompt):
    # Placeholder for chat interaction logic
    result = {"message": "Chat result based on the prompt and the DataFrame"}
    await asyncio.sleep(0)  # Simulate async work
    return result

async def main_viz():
    st.title("Excel Report Analysis")

    if os.path.exists(last_uploaded_file_path):
        st.success("File is available for visualization.")
        df = pd.read_csv(last_uploaded_file_path)
        col1, col2 = st.columns([1, 1])
        file_type = identify_file(df)
        st.write(file_type)
        with col1:
            st.dataframe(df, use_container_width=True)

        with col2:
            st.info("Chat Below")
            input_text = st.text_area("Enter your query")
            if input_text is not None:
                if st.button("Chat with CSV"):
                    st.info("Your Query: " + input_text)
                    result =await chat_with_csv(df, input_text)
                    st.success(result)
        if file_type == "3rd Party Sales Summary report":
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
                df = order_sales_summary_viz.preprocess_data(pd.read_csv(last_uploaded_file_path))
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
                low_stock_inventory_viz.low_stock_analysis_app(df)
                low_stock_inventory_viz.create_profit_margin_analysis_plot(df)
                low_stock_inventory_viz.create_low_stock_by_manufacturer_bar_plot(df)
            with cc2:
                low_stock_inventory_viz.create_interactive_price_vs_quantity_plot(df)
                low_stock_inventory_viz.create_quantity_price_ratio_plot(df)
        elif file_type == "Current Inventory report":
            cc1, cc2 = st.columns([1,1])

            with cc1:
                current_inventory_viz.df_analyze_inventory_value_by_category(df)
                current_inventory_viz.df_analyze_quantity_vs_retail_price(df)
                current_inventory_viz.df_analyze_inventory_value_by_manufacturer(df)
            with cc2:
                current_inventory_viz.df_analyze_inventory_value_per_unit(df)
                current_inventory_viz.df_compare_average_retail_prices(df)
        elif file_type == "Top Customers report":
            cc1, cc2 = st.columns([1,1])
            with cc1:
                top_customers_viz.customer_analysis_app(df)
                top_customers_viz.interactive_bar_plot_app(df)

            with cc2:
                top_customers_viz.create_non_zero_sales_grouped_plot(df)
                top_customers_viz.interactive_group_distribution_app(df)
        elif file_type == "Customer Details report":
            cc1, cc2 = st.columns([1,1])
            with cc1:
                customer_details_viz.plot_orders_and_sales_plotly(df)
                customer_details_viz.bar_plot_sorted_with_percentages(df)
            with cc2:
                customer_details_viz.create_interactive_non_zero_sales_plot(df)
                customer_details_viz.create_interactive_average_sales_heatmap(df)

        else:
                        #here can turn on lida and try to analyze dataset automatically by its toolset
            st.markdown('Description')
            st.caption('This is a string that explains something above.')

    else:
        st.warning("No file has been uploaded or downloaded yet.")

if __name__ == "__main__":
    asyncio.run(main_viz())
