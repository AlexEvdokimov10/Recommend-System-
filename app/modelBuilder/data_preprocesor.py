import pandas as pd


from app.dataPipline.data_pipline import PipelineContext, NormalizationStep, AdvancedDataCleaningStep, DataSavingStep, \
    DataBalancingStep, DimensionalityReductionStep, FeatureEngineeringStep, MatrixCreationStep, ClusteringStep, \
    BasicDataCleaningStep, DataSplittingStep


class DataPreprocessor:
    @staticmethod
    def clean(data: pd.DataFrame) -> pd.DataFrame:
        context = PipelineContext(data)
        BasicDataCleaningStep().process(context)
        AdvancedDataCleaningStep().process(context)
        return context.get_processed_data()

    @staticmethod
    def normalize(data: pd.DataFrame) -> pd.DataFrame:
        context = PipelineContext(data)
        NormalizationStep().process(context)
        return context.get_processed_data()

    @staticmethod
    def cluster(data: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
        context = PipelineContext(data)
        ClusteringStep(n_clusters=n_clusters).process(context)
        return context.get_processed_data()

    @staticmethod
    def create_similarity_matrix(data: pd.DataFrame) -> pd.DataFrame:
        context = PipelineContext(data)
        MatrixCreationStep().process(context)
        return context.get_processed_data()

    @staticmethod
    def feature_engineering(data: pd.DataFrame, feature_definitions: dict) -> pd.DataFrame:
        context = PipelineContext(data)
        FeatureEngineeringStep(feature_definitions=feature_definitions).process(context)
        return context.get_processed_data()

    @staticmethod
    def reduce_dimensions(data: pd.DataFrame, reduced_dim: int = 2) -> pd.DataFrame:
        context = PipelineContext(data)
        DimensionalityReductionStep(reduced_dim=reduced_dim).process(context)
        return context.get_processed_data()

    @staticmethod
    def balance_data(data: pd.DataFrame) -> pd.DataFrame:
        context = PipelineContext(data)
        DataBalancingStep().process(context)
        return context.get_processed_data()

    @staticmethod
    def save_data(data: pd.DataFrame, save_path: str) -> None:
        context = PipelineContext(data)
        DataSavingStep(save_path=save_path).process(context)

    @staticmethod
    def split(data: pd.DataFrame, split_type: str = "Holdout", test_size: float = 0.2, k: int = 5,
              stratified: bool = False, time_column: str = None):
        context = PipelineContext(data)
        splitting_step = DataSplittingStep(
            split_type=split_type,
            test_size=test_size,
            k=k,
            stratified=stratified,
            time_column=time_column
        )
        splitting_step.process(context)

        if split_type == "Holdout" or split_type == "Time-Based Split":
            return context.metadata.get("train_data"), context.metadata.get("test_data")
        elif split_type == "Cross-Validation":
            return context.metadata.get("cv_splits")
        else:
            raise ValueError(f"Unsupported split type: {split_type}")
