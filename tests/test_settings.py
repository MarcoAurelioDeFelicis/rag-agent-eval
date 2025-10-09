import os
from src import settings


def test_settings_values():
    # valori base e tipi
    assert isinstance(settings.CHUNK_SIZE, int)
    assert settings.CHUNK_SIZE > 0
    assert isinstance(settings.CHUNK_OVERLAP, int)
    assert isinstance(settings.EMBEDDING_MODEL_NAME, str)
    assert os.path.basename(settings.CSV_FILE_PATH) == "gz_recipe.csv"
