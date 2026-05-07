import streamlit as st
import pandas as pd
import plotly.express as px

from src.forecasting import create_forecast
from src.ai_assistant import ask_ai_question


st.set_page_config(
    page_title="AI Business Intelligence Platform",
    layout="wide"
)

st.title("AI-Powered Business Intelligence & Forecasting Platform")

st.write(
    "An interactive business intelligence platform for KPI tracking, sales analytics, forecasting, and AI-powered insights."
)


@st.cache_data
def load_data():
    return pd.read_csv("data/sales_data.csv")


df = load_data()

# -----------------------------
# Sidebar Filters
# -----------------------------

st.sidebar.header("Dashboard Filters")

filtered_df = df.copy()

if "Category" in df.columns:
    selected_categories = st.sidebar.multiselect(
        "Category",
        options=sorted(df["Category"].dropna().unique()),
        default=sorted(df["Category"].dropna().unique())
    )

    filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]

if "Region" in df.columns:
    selected_regions = st.sidebar.multiselect(
        "Region",
        options=sorted(df["Region"].dropna().unique()),
        default=sorted(df["Region"].dropna().unique())
    )

    filtered_df = filtered_df[filtered_df["Region"].isin(selected_regions)]

if "Segment" in df.columns:
    selected_segments = st.sidebar.multiselect(
        "Segment",
        options=sorted(df["Segment"].dropna().unique()),
        default=sorted(df["Segment"].dropna().unique())
    )

    filtered_df = filtered_df[filtered_df["Segment"].isin(selected_segments)]

date_columns = [
    col for col in df.columns
    if "date" in col.lower()
]

if len(date_columns) > 0:
    sidebar_date_column = date_columns[0]

    df[sidebar_date_column] = pd.to_datetime(
        df[sidebar_date_column],
        dayfirst=True,
        errors="coerce"
    )

    filtered_df[sidebar_date_column] = pd.to_datetime(
        filtered_df[sidebar_date_column],
        dayfirst=True,
        errors="coerce"
    )

    min_date = df[sidebar_date_column].min()
    max_date = df[sidebar_date_column].max()

    selected_date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date)
    )

    if len(selected_date_range) == 2:
        start_date, end_date = selected_date_range

        filtered_df = filtered_df[
            (filtered_df[sidebar_date_column] >= pd.to_datetime(start_date)) &
            (filtered_df[sidebar_date_column] <= pd.to_datetime(end_date))
        ]


numeric_columns = filtered_df.select_dtypes(include="number").columns.tolist()

# -----------------------------
# Tabs
# -----------------------------

overview_tab, visual_tab, forecast_tab, ai_tab, data_tab = st.tabs(
    [
        "Overview",
        "Visualizations",
        "Forecasting",
        "AI Assistant",
        "Data Preview"
    ]
)

# -----------------------------
# Overview Tab
# -----------------------------

with overview_tab:

    st.subheader("Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    if "Sales" in filtered_df.columns:
        total_sales = filtered_df["Sales"].sum()
        col1.metric("Total Sales", f"${total_sales:,.2f}")
    elif len(numeric_columns) > 0:
        col1.metric("Total Value", f"{filtered_df[numeric_columns[0]].sum():,.2f}")

    if "Profit" in filtered_df.columns:
        total_profit = filtered_df["Profit"].sum()
        col2.metric("Total Profit", f"${total_profit:,.2f}")
    elif len(numeric_columns) > 1:
        col2.metric("Secondary Value", f"{filtered_df[numeric_columns[1]].sum():,.2f}")

    if "Order ID" in filtered_df.columns:
        total_orders = filtered_df["Order ID"].nunique()
        col3.metric("Total Orders", f"{total_orders:,}")
    else:
        col3.metric("Rows Analyzed", f"{len(filtered_df):,}")

    if "Sales" in filtered_df.columns and "Order ID" in filtered_df.columns:
        avg_order_value = filtered_df["Sales"].sum() / max(filtered_df["Order ID"].nunique(), 1)
        col4.metric("Avg. Order Value", f"${avg_order_value:,.2f}")
    elif len(numeric_columns) > 0:
        col4.metric("Average Value", f"{filtered_df[numeric_columns[0]].mean():,.2f}")

    st.subheader("Business Summary")

    st.write(
        f"The dashboard is currently analyzing **{len(filtered_df):,} rows** after applying selected filters."
    )

    if "Category" in filtered_df.columns and "Sales" in filtered_df.columns:
        category_summary = (
            filtered_df.groupby("Category")["Sales"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        st.write("Top category by sales:")

        if not category_summary.empty:
            st.success(
                f"{category_summary.iloc[0]['Category']} generated the highest sales at ${category_summary.iloc[0]['Sales']:,.2f}."
            )

# -----------------------------
# Visualizations Tab
# -----------------------------

with visual_tab:

    st.subheader("Interactive Business Visualizations")

    if len(numeric_columns) > 0:
        selected_metric = st.selectbox(
            "Select Metric",
            numeric_columns
        )

        if sidebar_date_column in filtered_df.columns:
            trend_data = (
                filtered_df.groupby(sidebar_date_column)[selected_metric]
                .sum()
                .reset_index()
                .sort_values(sidebar_date_column)
            )

            trend_chart = px.line(
                trend_data,
                x=sidebar_date_column,
                y=selected_metric,
                title=f"{selected_metric} Trend Over Time"
            )

            st.plotly_chart(trend_chart, use_container_width=True)

        if "Category" in filtered_df.columns:
            category_chart_data = (
                filtered_df.groupby("Category")[selected_metric]
                .sum()
                .reset_index()
                .sort_values(selected_metric, ascending=False)
            )

            category_chart = px.bar(
                category_chart_data,
                x="Category",
                y=selected_metric,
                title=f"{selected_metric} by Category"
            )

            st.plotly_chart(category_chart, use_container_width=True)

        if "Region" in filtered_df.columns:
            region_chart_data = (
                filtered_df.groupby("Region")[selected_metric]
                .sum()
                .reset_index()
                .sort_values(selected_metric, ascending=False)
            )

            region_chart = px.bar(
                region_chart_data,
                x="Region",
                y=selected_metric,
                title=f"{selected_metric} by Region"
            )

            st.plotly_chart(region_chart, use_container_width=True)

# -----------------------------
# Forecasting Tab
# -----------------------------

with forecast_tab:

    st.subheader("Machine Learning Forecasting Model")

    if len(date_columns) > 0 and len(numeric_columns) > 0:

        date_column = st.selectbox(
            "Select Date Column for Forecasting",
            date_columns
        )

        target_column = st.selectbox(
            "Select Target Column to Forecast",
            numeric_columns
        )

        forecast_days = st.slider(
            "Forecast Days",
            min_value=7,
            max_value=90,
            value=30
        )

        try:
            historical_data, forecast_data = create_forecast(
                filtered_df,
                date_column,
                target_column,
                forecast_days
            )

            historical_chart = px.line(
                historical_data,
                x=date_column,
                y=target_column,
                title=f"Historical {target_column} Over Time"
            )

            forecast_chart = px.line(
                forecast_data,
                x=date_column,
                y="Forecast",
                title=f"{forecast_days}-Day {target_column} Forecast"
            )

            st.plotly_chart(historical_chart, use_container_width=True)
            st.plotly_chart(forecast_chart, use_container_width=True)

        except Exception as e:
            st.warning(
                f"Forecasting could not be completed. Please check your selected columns. Error: {e}"
            )

    else:
        st.info("Forecasting requires at least one date column and one numeric column.")

# -----------------------------
# AI Assistant Tab
# -----------------------------

with ai_tab:

    st.subheader("AI Business Intelligence Assistant")

    st.write(
        "Ask natural language questions about business performance, trends, KPIs, and forecasting insights."
    )

    user_question = st.text_input(
        "Ask a business question",
        placeholder="Example: What are the strongest business trends in this dataset?"
    )

    if st.button("Ask AI Assistant"):

        if user_question.strip() == "":
            st.warning("Please enter a question first.")

        else:
            with st.spinner("Analyzing dataset..."):
                try:
                    ai_response = ask_ai_question(filtered_df, user_question)
                    st.write(ai_response)

                except Exception as e:
                    st.error(f"AI assistant error: {e}")

# -----------------------------
# Data Preview Tab
# -----------------------------

with data_tab:

    st.subheader("Filtered Dataset Preview")

    st.dataframe(filtered_df.head(100))

    st.write(f"Showing preview from **{len(filtered_df):,} filtered rows**.")