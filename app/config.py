import sys
import os
from dotenv import load_dotenv
import logging
from app.utils.whatsapp_utils import setup_llm_config

def load_configurations(app):
    load_dotenv()
    app.config["ACCESS_TOKEN"] = os.getenv("ACCESS_TOKEN")
    app.config["YOUR_PHONE_NUMBER"] = os.getenv("YOUR_PHONE_NUMBER")
    app.config["APP_ID"] = os.getenv("APP_ID")
    app.config["APP_SECRET"] = os.getenv("APP_SECRET")
    app.config["RECIPIENT_WAID"] = os.getenv("RECIPIENT_WAID")
    app.config["VERSION"] = os.getenv("VERSION")
    app.config["PHONE_NUMBER_ID"] = os.getenv("PHONE_NUMBER_ID")
    app.config["VERIFY_TOKEN"] = os.getenv("VERIFY_TOKEN")
    app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    app.config['LLM_PROVIDER'] = os.getenv('LLM_PROVIDER', 'openai')
    app.config['OPENAI_MODEL'] = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    app.config['MAX_TOKENS'] = int(os.getenv('MAX_TOKENS', 500))
    app.config['TEMPERATURE'] = float(os.getenv('TEMPERATURE', 0.7))
    app.config['SEARCH_TOP_K'] = int(os.getenv('SEARCH_TOP_K', 3))
    app.config['SIMILARITY_THRESHOLD'] = float(os.getenv('SIMILARITY_THRESHOLD', 0.3))




def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
