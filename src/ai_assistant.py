import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def summarize_dataset(df):

    columns = df.columns.tolist()

    row_count = len(df)

    numeric_summary = df.describe().to_string()

    return f"""
Dataset columns: {columns}

Number of rows: {row_count}

Numeric summary:
{numeric_summary}
"""


def ask_ai_question(df, question):

    dataset_summary = summarize_dataset(df)

    prompt = f"""
You are a business intelligence analyst.

Use the dataset summary below to answer the user's business question clearly and professionally.

Focus on:
- business trends
- KPIs
- revenue insights
- forecasting interpretation
- operational analysis

Dataset Summary:
{dataset_summary}

User Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a professional business intelligence and analytics assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content