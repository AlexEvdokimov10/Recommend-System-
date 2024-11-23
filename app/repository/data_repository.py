from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import psycopg2
from psycopg2 import sql
from ..utils import get_database_url

DATABASE_URL = get_database_url('dev')

class Data:
    def __init__(self, **kwargs):
         for key, value in kwargs.items():
            setattr(self, key, value)

class DataSource(ABC):
    def __init__(self, source_id: str, source_type: str):
        self.source_id = source_id
        self.source_type = source_type

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def execute_query(self, query: str, params: Optional[tuple] = None, fetch: bool = False) ->  Optional[List[tuple]]:
        pass


class DatabaseSource(DataSource):
    def __init__(self, source_id: str):
        super().__init__(source_id, "Database")
        self.connection = None

    def connect(self):
        if not self.connection:
            self.connection = psycopg2.connect(DATABASE_URL)
            print("Connected to database")

    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            print("Disconnected from database")

  

    def save_dataframe(self, df: pd.DataFrame, table_name: str):
        
        with self.connection.cursor() as cursor:
            columns = ", ".join([f"{col} VARCHAR" for col in df.columns])
            cursor.execute(sql.SQL(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"))
            self.connection.commit()

            total_rows = len(df)
            
            if total_rows == 0:
                raise ValueError("The DataFrame is empty and cannot be saved to the database.")

            for i, row in enumerate(df.iterrows(), start=1):
                cursor.execute(
                    sql.SQL(f"INSERT INTO {table_name} VALUES ({','.join(['%s'] * len(row[1]))})"),
                    tuple(row[1])
                )

                progress = (i / total_rows) * 100
                print(f"Progress: {progress:.2f}% ({i}/{total_rows} rows inserted)", end="\r")

            self.connection.commit()
            
    def execute_query(self, query: str, params: Optional[tuple] = None, fetch: bool = False) -> Optional[List[tuple]]:
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()  
            else:
                self.connection.commit()  
        return None
    

    def load_csv_to_db(self, csv_file_path: str, table_name: str):
    
        try:
            df = pd.read_csv(csv_file_path)
            if df.empty:
                raise ValueError(f"The file '{csv_file_path}' is empty.")
            self.connect()
            self.save_dataframe(df, table_name)
            print(f"Data from '{csv_file_path}' successfully loaded into table '{table_name}'.")
        except FileNotFoundError:
            print(f"File '{csv_file_path}' not found.")
            raise
        except Exception as e:
            print(f"An error occurred while loading the CSV file: {e}")
            raise
        finally:
            self.disconnect()


class DataMapper:
    def map_to_domain(self, raw_data: tuple, column_names: List[str]) -> Data:
        data_dict = dict(zip(column_names, raw_data))
        return Data(**data_dict)

    def map_to_storage(self, data: Data) -> Dict[str, Any]:
        return data.__dict__


class DataRepository:
    def __init__(self, data_source: DataSource, mapper: DataMapper):
        self.data_source = data_source
        self.mapper = mapper

    def get_data(self, table_name: str) -> List[Data]:
        self.data_source.connect()
        try:
            query_columns = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}';"
            columns = self.data_source.execute_query(query_columns, fetch=True)
            column_names = [col[0] for col in columns]

            query_data = f"SELECT * FROM {table_name};"
            raw_data = self.data_source.execute_query(query_data, fetch=True)

            return [self.mapper.map_to_domain(row, column_names) for row in raw_data] if raw_data else []
        finally:
            self.data_source.disconnect()

    def save_dataframe(self, df: pd.DataFrame, table_name: str):

        self.data_source.connect()
        try:
            
            column_definitions = ", ".join([f"{col} TEXT" for col in df.columns])
            create_table_query = sql.SQL(f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions});")
            self.data_source.execute_query(create_table_query)

            insert_query = sql.SQL(
                f"INSERT INTO {table_name} VALUES ({','.join(['%s'] * len(df.columns))})"
            )
            for _, row in df.iterrows():
                self.data_source.execute_query(insert_query, tuple(row))
        finally:
            self.data_source.disconnect()

    def get_table_columns(self, table_name: str) -> List[Dict[str, str]]:

        self.data_source.connect()
        try:
            query = """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s;
            """
            result = self.data_source.execute_query(query, params=(table_name,), fetch=True)
            return [{"column_name": row[0], "data_type": row[1]} for row in result]
        finally:
            self.data_source.disconnect()