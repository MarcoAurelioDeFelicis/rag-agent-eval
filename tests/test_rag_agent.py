import importlib
import sys
from unittest import mock


def test_create_rag_agent_builds_chain(monkeypatch):
    # Mock external packages before import
    sys.modules.setdefault('langchain_google_genai', mock.MagicMock())
    sys.modules.setdefault('langchain_core.prompts', mock.MagicMock())
    sys.modules.setdefault('langchain.chains.combine_documents', mock.MagicMock())
    sys.modules.setdefault('langchain.chains.retrieval', mock.MagicMock())

    ra = importlib.import_module("src.rag_agent")

    fake_llm = object()
    monkeypatch.setattr(sys.modules['langchain_google_genai'], "ChatGoogleGenerativeAI", lambda model, temperature: fake_llm)
    monkeypatch.setattr(ra, "ChatGoogleGenerativeAI", lambda model, temperature: fake_llm)

    # mock prompt template and chain builders
    fake_prompt = object()
    # ChatPromptTemplate is imported from langchain_core.prompts inside module; ensure attribute
    sys.modules['langchain_core.prompts'].ChatPromptTemplate = mock.MagicMock()
    monkeypatch.setattr(ra.ChatPromptTemplate, "from_template", lambda t: fake_prompt)

    fake_doc_chain = object()
    monkeypatch.setattr(ra, "create_stuff_documents_chain", lambda llm, prompt: fake_doc_chain)
    fake_retriever = mock.MagicMock()

    class FakeDB:
        def as_retriever(self):
            return fake_retriever

    fake_retrieval_chain = object()
    monkeypatch.setattr(ra, "create_retrieval_chain", lambda retriever, doc_chain: fake_retrieval_chain)

    result = ra.create_rag_agent(FakeDB(), model_name="mymodel")
    assert result is fake_retrieval_chain
