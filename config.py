import os
from dotenv import load_dotenv


load_dotenv()

class Config:

    ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev')
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')
    DATABASE_URL = os.getenv(f'DATABASE_URL_{ENVIRONMENT.upper()}')

    SAVE_TEMPLATE_TO_DB = os.getenv('SAVE_TEMPALE_TO_DB', 'False').lower() == 'true'
    SAVE_TEMPLATE_AS_FILE = os.getenv('SAVE_TEMPATE_AS_FILE', 'False').lower() == 'true'
    DELETE_TEMPLATE_TO_DB = os.getenv('DELETE_TEMPALE_TO_DB', 'False').lower() == 'true'
    DELETE_TEMPLATE_AS_FILE = os.getenv('DELETE_TEMPATE_AS_FILE', 'False').lower() == 'true'


    DEBUG = ENVIRONMENT == 'dev'
    TESTING = ENVIRONMENT == 'test'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG' if DEBUG else 'INFO')


    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    JSON_SORT_KEYS = os.getenv('JSON_SORT_KEYS', 'True').lower() == 'true'


    CACHE_TYPE = os.getenv('CACHE_TYPE', 'SimpleCache')
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_DEFAULT_TIMEOUT', 300))


    SESSION_COOKIE_SECURE = os.getenv('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    REMEMBER_COOKIE_DURATION = int(os.getenv('REMEMBER_COOKIE_DURATION', 3600))

class DevelopmentConfig(Config):


    DEBUG = True
    DATABASE_URL = os.getenv('DATABASE_URL_DEV')
    ENABLE_DEBUG_TOOLBAR = os.getenv('ENABLE_DEBUG_TOOLBAR', 'True').lower() == 'true'

class TestingConfig(Config):


    TESTING = True
    DATABASE_URL = os.getenv('DATABASE_URL_TEST')
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):

    DEBUG = False
    TESTING = False
    DATABASE_URL = os.getenv('DATABASE_URL_PROD')
    SESSION_COOKIE_SECURE = True
    LOG_LEVEL = 'WARNING'
    JSON_SORT_KEYS = False