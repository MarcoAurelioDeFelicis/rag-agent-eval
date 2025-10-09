import importlib
import sys
from unittest import mock


def test_get_accuracy_evaluator_calls_load_evaluator(monkeypatch):
    # Mock external modules before importing the target module
    if 'langchain' not in sys.modules:
        sys.modules['langchain'] = mock.MagicMock()
    # langchain.evaluation.load_evaluator
    sys.modules['langchain.evaluation'] = mock.MagicMock()
    # langchain_google_genai.ChatGoogleGenerativeAI
    sys.modules['langchain_google_genai'] = mock.MagicMock()

    # Now import the module under test
    ev = importlib.import_module("src.evaluator")

    # Prepare fake llm and fake evaluator
    fake_llm = mock.MagicMock(name="fake_llm")
    # Patch the symbol inside the imported module so it's used by the function
    monkeypatch.setattr(ev, "ChatGoogleGenerativeAI", lambda model, temperature, model_kwargs=None: fake_llm)

    fake_evaluator = object()

    def fake_load_evaluator(name, criteria, llm):
        assert llm is fake_llm
        return fake_evaluator

    # ensure the evaluator module uses our fake implementation
    monkeypatch.setattr(ev, "load_evaluator", fake_load_evaluator)

    # call the function
    res = ev.get_accuracy_evaluator("some-model")
    assert res is fake_evaluator
