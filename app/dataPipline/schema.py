import graphene
from app.repository.data_repository import DataRepository, DatabaseSource, DataMapper
from .utils import default_process_data, save_processed_data
import pandas as pd

data_source = DatabaseSource("db_source")
data_mapper = DataMapper()
data_repo = DataRepository(data_source, data_mapper)


default_steps = ['data_cleaning_step', 'normalization_step', 'clustering_step', 'matrix_creation_step', 'feature_engineering_step',
                 'dimensionality_reduction_step']
is_save_processed_data = True


class ProcessDataMutation(graphene.Mutation):
    class Arguments:
        table_name = graphene.String(required=True)
        steps = graphene.List(graphene.String, required=False)

    processed_data = graphene.JSONString()
    success = graphene.Boolean()
    message = graphene.String()

    def mutate(self, info, table_name, steps=None):
        try:
            steps = steps or default_steps

            raw_data = data_repo.get_data(table_name)
            if not raw_data:
                return ProcessDataMutation(success=False, message=f"No data found in table '{table_name}'", processed_data=None)


            processed_data = default_process_data(raw_data, steps)


            if is_save_processed_data:
                save_processed_data(processed_data, 'processed_data_anime_frame')

            return ProcessDataMutation(success=True, message="Data processed successfully", processed_data=processed_data.tolist())

        except Exception as e:
            return ProcessDataMutation(success=False, message=f"Error processing data: {str(e)}", processed_data=None)


class Mutation(graphene.ObjectType):
    process_data = ProcessDataMutation.Field()


def create_schema_pipline():
    schema = graphene.Schema(mutation=Mutation)
    try:
        print("[INFO] Schema created successfully.")
        return schema
    except Exception as e:
        print(f"[ERROR] {e}")
        return None