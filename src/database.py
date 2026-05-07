import sqlite3
import pandas as pd


DATABASE_NAME = "business_data.db"


def create_connection():
    connection = sqlite3.connect(DATABASE_NAME)
    return connection


def load_data_to_database(df, table_name="sales_data"):

    connection = create_connection()

    df.to_sql(
        table_name,
        connection,
        if_exists="replace",
        index=False
    )

    connection.close()


def run_query(query):

    connection = create_connection()

    result = pd.read_sql_query(
        query,
        connection
    )

    connection.close()

    return result