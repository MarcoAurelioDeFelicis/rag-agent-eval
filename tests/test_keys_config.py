import os
import importlib
import types

import pytest


def test_configure_api_keys_missing(monkeypatch):
    # Simula .env senza chiavi
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "")

    # ricarica il modulo per eseguire la funzione
    keys = importlib.import_module("src.keys_config")

    with pytest.raises(ValueError):
        keys.configure_api_keys()
