import pandas as pd

from .data_pipline import (
    DataPipeline,
    PipelineContext,
    NormalizationStep,
    ClusteringStep,
    MatrixCreationStep,
    FeatureEngineeringStep,
    DimensionalityReductionStep, BasicDataCleaningStep, AdvancedDataCleaningStep, DataSplittingStep,
)
from ..repository.data_repository import DatabaseSource


def default_process_data(raw_data, steps, split_params=None):

    df = pd.DataFrame([item.__dict__ for item in raw_data])
    print(f"Raw DataFrame:\n{df}")

    if df.empty:
        print("[default_process_data] DataFrame is empty. Returning empty list.")
        return {"error": "Data is empty after transformation", "processed_data": []}

    context = PipelineContext(df)
    pipeline = DataPipeline()

    for step in steps:
        if step == "data_basic_cleaning_step":
            pipeline.add_step(BasicDataCleaningStep())
        elif step == "data_advanced_cleaning_step":
            pipeline.add_step(AdvancedDataCleaningStep())
        elif step == "normalization_step":
            pipeline.add_step(NormalizationStep())
        elif step == "clustering_step":
            pipeline.add_step(ClusteringStep(n_clusters=5))
        elif step == "matrix_creation_step":
            pipeline.add_step(MatrixCreationStep())
        elif step == "feature_engineering_step":
            feature_definitions = {
                'weighted_rating': ['episodes', 'rating'],
                'interaction_score': ['rating', 'members'],
                'normalized_members': ['members']
            }
            feature_engineering_step = FeatureEngineeringStep(feature_definitions)
            pipeline.add_step(feature_engineering_step)
        elif step == "dimensionality_reduction_step":
            pipeline.add_step(DimensionalityReductionStep(reduced_dim=5))
        elif step == "data_splitting_step":
            if not split_params:
                raise ValueError("split_params must be provided for data_splitting_step.")
            split_type = split_params.get("split_type", "Holdout")
            test_size = split_params.get("test_size", 0.2)
            k = split_params.get("k", 5)
            stratified = split_params.get("stratified", False)
            time_column = split_params.get("time_column", None)
            pipeline.add_step(DataSplittingStep(split_type, test_size, k, stratified, time_column))

    pipeline.execute(context)

    if "data_splitting_step" in steps:
        return {
            "train_data": context.metadata.get("train_data"),
            "test_data": context.metadata.get("test_data"),
            "cv_splits": context.metadata.get("cv_splits"),
        }

    processed_data = context.get_processed_data()
    return processed_data


def save_processed_data(processed_data, table_name):
    df = pd.DataFrame(processed_data)
    if not df.empty:
        data_source = DatabaseSource("db_source")
        data_source.connect()
        try:
            data_source.save_dataframe(df, table_name)
            print(f"Processed data successfully saved to table '{table_name}'.")
        finally:
            data_source.disconnect()
    else:
        print("No processed data to save.")
