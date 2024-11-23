import os

from flask import Flask
from flask_graphql import GraphQLView

from config import DevelopmentConfig, TestingConfig, ProductionConfig
from .dataPipline.main import data_pipeline_bp
from .dataPipline.schema import create_schema_pipline
from .modelBuilder.main import model_builder_bp
from .modelBuilder.schema import create_model_builder_schema
from .repository.main import rest_bp
from .repository.main import rest_bp
from .repository.schema import create_schema_repository
from .templateManager.main import tmp_bp
from .visualizationManager.main import visualization_bp


def create_app():
    app = Flask(__name__)

    env = os.getenv('ENVIRONMENT', 'dev')
    if env == 'dev':
        app.config.from_object(DevelopmentConfig)
    elif env == 'test':
        app.config.from_object(TestingConfig)
    elif env == 'prod':
        app.config.from_object(ProductionConfig)
    else:
        raise ValueError(f"Unknown environment: {env}")

    if os.getenv('ENABLE_GRAPHQL', 'False').lower() == 'true':
        app.add_url_rule(
            "/graphql",
            '/repository',
            view_func=GraphQLView.as_view(
                "graphql_repository",
                schema=create_schema_repository(),
                graphiql=True,
            ),
        )
        app.add_url_rule(
            "/graphql",
            '/pipeline',
            view_func=GraphQLView.as_view(
                "graphql_pipeline",
                schema=create_schema_pipline(),
                graphiql=True,
            ),
        )

        app.add_url_rule(
            "/graphql/model_builder",
            view_func=GraphQLView.as_view(
                "graphql_model_builder",
                schema=create_model_builder_schema(),
                graphiql=True,
            ),
        )

    if os.getenv('ENABLE_REST', 'False').lower() == 'true':
        app.register_blueprint(rest_bp)
        app.register_blueprint(data_pipeline_bp, url_prefix="/data_pipeline")
        app.register_blueprint(model_builder_bp, url_prefix="/model_builder")
        app.register_blueprint(visualization_bp)
        app.register_blueprint(tmp_bp)

    return app
