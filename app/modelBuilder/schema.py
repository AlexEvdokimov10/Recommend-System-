import graphene

from .data_preprocesor import DataPreprocessor
from .model_parameters import ModelParameters
from .modelBuilder import ModelBuilder

from .model_strategy import NeuralNetworkCollaborativeFiltering, ContentBasedFiltering, ModelBasedFiltering, \
    MovieRecommendationStrategy, CollaborativeFiltering
from .training_data import TrainingData
import pandas as pd

from ..repository.data_repository import DatabaseSource, DataRepository, DataMapper
from ..utils import to_dataframe

data_source = DatabaseSource("db_source")
data_mapper = DataMapper()
data_repo = DataRepository(data_source, data_mapper)


class TrainModelMutation(graphene.Mutation):
    class Arguments:
        table_name = graphene.String(required=True)
        strategy = graphene.String(required=True)
        parameters = graphene.JSONString()

    success = graphene.Boolean()
    message = graphene.String()

    def mutate(self, info, table_name, strategy, parameters=None):
        try:
            parameters = parameters or {}

            raw_data = data_repo.get_data(table_name)
            if not raw_data:
                raise ValueError(f"No data found in table '{table_name}'")

            data = to_dataframe(raw_data)

            clean_data = DataPreprocessor.clean(data)
            normalized_data = DataPreprocessor.normalize(clean_data)
            train_data, _ = DataPreprocessor.split(normalized_data)

            features = train_data.drop(columns=['label'])
            labels = train_data['label']
            training_data = TrainingData(features, labels)

            model_params = ModelParameters(
                learning_rate=parameters.get("learning_rate", 0.01),
                max_iterations=parameters.get("max_iterations", 100),
                batch_size=parameters.get("batch_size", 32)
            )

            strategies = {
                "CollaborativeFiltering": CollaborativeFiltering,
                "ContentBasedFiltering": ContentBasedFiltering,
                "NeuralNetworkCollaborativeFiltering": NeuralNetworkCollaborativeFiltering,
                "MovieRecommendationStrategy":MovieRecommendationStrategy,
                "ModelBasedFiltering": ModelBasedFiltering
            }
            strategy_class = strategies.get(strategy)
            if not strategy_class:
                raise ValueError(f"Unknown strategy: {strategy}")

            strategy_instance = strategy_class()


            model_builder = ModelBuilder(model_params)
            model_builder.set_strategy(strategy_instance)
            model_builder.build_model(training_data)

            return TrainModelMutation(success=True, message="Model trained successfully.")
        except Exception as e:
            return TrainModelMutation(success=False, message=str(e))


class Query(graphene.ObjectType):
    ping = graphene.String(default_value="ModelBuilder API is running")


class Mutation(graphene.ObjectType):
    train_model = TrainModelMutation.Field()


def create_model_builder_schema():
    return graphene.Schema(query=Query, mutation=Mutation)