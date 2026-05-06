import streamlit as st
import pandas as pd
import plotly.express as px
from src.forecasting import create_forecast

st.set_page_config(
    page_title="AI Business Intelligence Platform",
    layout="wide"
)

st.title("AI-Powered Business Intelligence & Forecasting Platform")

st.write(
    "Interactive analytics dashboard for KPI tracking, forecasting, and business intelligence insights."
)

df = pd.read_csv("data/sales_data.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

numeric_columns = df.select_dtypes(include='number').columns.tolist()

st.subheader("Key Business Metrics")

col1, col2, col3 = st.columns(3)

if len(numeric_columns) > 0:

    primary_metric = numeric_columns[0]

    total_value = df[primary_metric].sum()
    average_value = df[primary_metric].mean()

    col1.metric(
        "Total Value",
        f"{total_value:,.2f}"
    )

    col2.metric(
        "Average Value",
        f"{average_value:,.2f}"
    )

    col3.metric(
        "Rows Analyzed",
        len(df)
    )

st.subheader("Data Visualization")

selected_column = st.selectbox(
    "Select Numeric Column",
    numeric_columns
)

fig = px.histogram(
    df,
    x=selected_column,
    title=f"{selected_column} Distribution",
)

st.plotly_chart(
    fig,
    use_container_width=True
)

if "Category" in df.columns and len(numeric_columns) > 0:

    category_chart = px.bar(
        df.groupby("Category")[selected_column]
        .sum()
        .reset_index(),
        x="Category",
        y=selected_column,
        title=f"{selected_column} by Category"
    )

    st.plotly_chart(
        category_chart,
        use_container_width=True
    )
st.subheader("Sales Forecasting Model")

date_columns = [
    col for col in df.columns
    if "date" in col.lower()
]

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
            df,
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