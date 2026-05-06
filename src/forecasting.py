import pandas as pd
from sklearn.linear_model import LinearRegression


def prepare_forecast_data(df, date_column, target_column):
    df = df.copy()

    df[date_column] = pd.to_datetime(
    df[date_column],
    dayfirst=True
)

    daily_data = (
        df.groupby(date_column)[target_column]
        .sum()
        .reset_index()
        .sort_values(date_column)
    )

    daily_data["day_number"] = range(len(daily_data))

    return daily_data


def create_forecast(df, date_column, target_column, forecast_days=30):
    daily_data = prepare_forecast_data(df, date_column, target_column)

    X = daily_data[["day_number"]]
    y = daily_data[target_column]

    model = LinearRegression()
    model.fit(X, y)

    future_days = pd.DataFrame({
        "day_number": range(len(daily_data), len(daily_data) + forecast_days)
    })

    predictions = model.predict(future_days)

    last_date = daily_data[date_column].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days
    )

    forecast_df = pd.DataFrame({
        date_column: future_dates,
        "Forecast": predictions
    })

    return daily_data, forecast_df