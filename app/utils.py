import os

import pandas as pd
from dotenv import load_dotenv
from graphql import GraphQLError

load_dotenv()

def get_database_url(env: str) -> str:
    url = os.getenv(f'DATABASE_URL_{env.upper()}')
    if not url:
        raise ValueError(f"Database URL for environment '{env}' is not set.")
    return url


def format_error(error):

    if isinstance(error, GraphQLError):

        return {
            "message": error.message,
            "locations": [
                {"line": loc.line, "column": loc.column} for loc in error.locations or []
            ],
            "path": error.path,
        }
    return {"message": str(error)}

def to_dataframe(raw_data):
    if not raw_data:
        raise ValueError("Raw data is empty and cannot be converted to a DataFrame.")
    return pd.DataFrame([item.__dict__ for item in raw_data])
