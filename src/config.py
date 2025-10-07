import os
from dotenv import load_dotenv

""" Load of the google api key to set it up as env variable """

def configure_api_key():

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ ERROR: NO API key founded!")
    os.environ["GOOGLE_API_KEY"] = api_key