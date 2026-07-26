import pytest
from fastapi.testclient import TestClient

import main as m
from main import app

client = TestClient(app)


# --- Fake replacements for the heavy/networked parts ---

class FakeTokenizer:
    def encode(self, text):
        return [text]

    def convert_ids_to_tokens(self, ids):
        return ids

    def convert_tokens_to_ids(self, tokens):
        return tokens

    def decode(self, ids):
        return ids[0] if ids else ""


class FakeTranslator:
    call_count = 0

    def translate_batch(self, batch):
        FakeTranslator.call_count += 1

        class Result:
            def __init__(self, tokens):
                self.hypotheses = [tokens]

        return [Result([f"TRANSLATED({token})" for token in tokens]) for tokens in batch]


@pytest.fixture(autouse=True)
def mock_heavy_dependencies(monkeypatch):
    """Runs automatically before every test in this file. Replaces model
    downloading/loading with fast, deterministic fakes, and resets all
    caches so tests don't leak state into one another."""

    monkeypatch.setattr(m, "ensure_ct2_model", lambda model_name, ct2_dir: None)
    monkeypatch.setattr(
        m.AutoTokenizer, "from_pretrained", staticmethod(lambda name: FakeTokenizer())
    )
    monkeypatch.setattr(m.ctranslate2, "Translator", lambda *args, **kwargs: FakeTranslator())

    m.loaded_models.clear()
    m.translation_cache.clear()
    m.semantic_cache.clear()
    FakeTranslator.call_count = 0

    yield


# --- Tests ---

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_translate_unsupported_language_pair():
    response = client.post("/translate", json={
        "text": "Hello",
        "source_lang": "en",
        "target_lang": "xx",
    })
    assert response.status_code == 400


def test_translate_success():
    response = client.post("/translate", json={
        "text": "Hello",
        "source_lang": "en",
        "target_lang": "pt",
    })
    assert response.status_code == 200
    assert "TRANSLATED" in response.json()["translation"]


def test_glossary_exact_match_skips_translation_model():
    response = client.post("/translate", json={
        "text": "Transdom",
        "source_lang": "en",
        "target_lang": "pt",
    })
    assert response.status_code == 200
    # If the glossary worked, the fake translator was never called —
    # so the output should NOT be wrapped in "TRANSLATED(...)".
    assert response.json()["translation"] == "Transdom"


def test_glossary_partial_match_is_masked_and_restored():
    response = client.post("/translate", json={
        "text": "Please click Login to continue",
        "source_lang": "en",
        "target_lang": "pt",
    })
    assert response.status_code == 200
    translation = response.json()["translation"]
    # The glossary's custom translation for "Login" should appear in the
    # output, even though the text went through the (fake) translation model.
    assert "Entrar" in translation


def test_batch_rejects_too_many_texts():
    too_many = ["hello"] * (m.MAX_TEXTS_PER_BATCH + 1)
    response = client.post("/translate/batch", json={
        "texts": too_many,
        "source_lang": "en",
        "target_lang": "pt",
    })
    assert response.status_code == 422


def test_rejects_text_over_max_length():
    too_long = "a" * (m.MAX_TEXT_LENGTH + 1)
    response = client.post("/translate", json={
        "text": too_long,
        "source_lang": "en",
        "target_lang": "pt",
    })
    assert response.status_code == 422


def test_lru_evicts_least_recently_used_model(monkeypatch):
    monkeypatch.setattr(m, "MAX_LOADED_MODELS", 1)

    client.post("/translate", json={"text": "a", "source_lang": "en", "target_lang": "pt"})
    client.post("/translate", json={"text": "b", "source_lang": "en", "target_lang": "es"})

    assert len(m.loaded_models) == 1
    assert ("en", "es") in m.loaded_models
    assert ("en", "pt") not in m.loaded_models


def test_translation_cache_avoids_recomputation():
    payload = {"text": "Repeated sentence", "source_lang": "en", "target_lang": "pt"}

    client.post("/translate", json=payload)
    calls_after_first = FakeTranslator.call_count

    client.post("/translate", json=payload)
    calls_after_second = FakeTranslator.call_count

    assert calls_after_second == calls_after_first