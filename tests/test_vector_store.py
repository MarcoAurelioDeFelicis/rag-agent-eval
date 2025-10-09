import os
import tempfile
import csv
import importlib
import sys
from unittest import mock


def make_sample_csv(path):
    with open(path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text"])
        writer.writerow(["1", "This is a test recipe."])


def test_create_vector_store_creates_and_loads(tmp_path, monkeypatch):
    # crea un csv temporaneo
    csv_file = tmp_path / "sample.csv"
    make_sample_csv(str(csv_file))

    # mock settings per usare il csv creato
    settings = importlib.import_module("src.settings")
    monkeypatch.setattr(settings, "CSV_FILE_PATH", str(csv_file))

    # mock dei pacchetti esterni usati dal modulo per evitare import pesanti
    mock_modules = [
        'langchain_community',
        'langchain_community.document_loaders',
        'langchain_community.document_loaders.csv_loader',
        'langchain_text_splitters',
        'langchain_huggingface',
        'langchain_community.vectorstores',
    ]

    for name in mock_modules:
        if name not in sys.modules:
            sys.modules[name] = mock.MagicMock()

    # mock degli embeddings e FAISS per evitare chiamate pesanti
    fake_faiss = mock.MagicMock()
    sys.modules['langchain_community.vectorstores'].FAISS = fake_faiss

    # Importa il modulo da testare dopo i mock
    vs = importlib.import_module("src.vector_store")

    # Mock FAISS nel modulo importato
    monkeypatch.setattr(vs, "FAISS", fake_faiss)
    fake_faiss.load_local.return_value = fake_faiss
    fake_faiss.from_documents.return_value = fake_faiss

    # invoca la funzione
    db = vs.create_vector_store(str(csv_file), persist_directory=str(tmp_path / "db"))

    # controlli minimi
    assert db is not None
    assert fake_faiss.from_documents.called
