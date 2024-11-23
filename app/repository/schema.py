from graphene import ObjectType, Int, String, Float, List as GraphQLList, Schema
import pandas as pd
from app.repository.data_repository import DataRepository, DatabaseSource, DataMapper
from .types import AnimeType, VALID_TABLES, VALID_SERIALIZERS

data_source = DatabaseSource("db_source")
data_mapper = DataMapper()
data_repo = DataRepository(data_source, data_mapper)




class Query(ObjectType):

    all_data = GraphQLList(
        lambda: AnimeType,
        table_name=String(required=True),
        serializer=String(required=True),
    )

    def resolve_all_data(root, info, table_name, serializer):
        print(f"Resolving data for table: {table_name}")

        if table_name not in VALID_TABLES:
            raise ValueError(f"Table '{table_name}' was not found in VALID_TABLES.")
        if serializer not in VALID_SERIALIZERS:
            raise ValueError(f"Serializer '{serializer}' was not found in VALID_SERIALIZERS.")

        query = f"SELECT * FROM {table_name};"
        print(f"Query: {query}")

        df = load_data_as_dataframe(query)

        if df.empty:
            raise ValueError(f"No data found in table '{table_name}'.")


        serializer_class = VALID_SERIALIZERS[serializer]
        return [serializer_class(**row) for row in df.to_dict(orient="records")]


def load_data_as_dataframe(query):

    data_source.connect()
    try:

        result = data_source.execute_query(query, fetch=True)


        table_name = query.split("FROM")[1].strip().split()[0].rstrip(";")


        if table_name not in VALID_TABLES:
            raise ValueError(f"Table '{table_name}' is not a valid table.")


        columns_query = f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}';
        """
        print(f"Columns Query: {columns_query}")
        columns_result = data_source.execute_query(columns_query, fetch=True)


        if not columns_result:
            raise ValueError(f"No columns found for table '{table_name}'.")
        if not result:
            raise ValueError(f"No data found for query: {query}")


        columns = [col[0] for col in columns_result]


        df = pd.DataFrame(result, columns=columns)
        return df
    finally:
        data_source.disconnect()

def create_schema_repository():
    try:
        schema = Schema(query=Query)
        print("[INFO] Schema created successfully.")
        return schema
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


