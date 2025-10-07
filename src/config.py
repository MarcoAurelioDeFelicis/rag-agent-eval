import os
from dotenv import load_dotenv

""" Load of the google api key to set it up as env variable """

def configure_api_keys():

    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("❌ ERROR: NO google API key founded!")
    os.environ["GOOGLE_API_KEY"] = google_api_key

    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_api_token:
        raise ValueError("❌ ERROR: NO hugging face API key founded!")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token