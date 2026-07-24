import os
from collections import OrderedDict
from typing import Annotated

import ctranslate2
from dotenv import load_dotenv
from ctranslate2.converters import TransformersConverter
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# Load variables from .env into the environment. Falls back to safe
# default if a variable isn't set, so the server still runs without
# a .env file (useful for quick local testing).
load_dotenv()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5000").split(",")
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")
MAX_TEXTS_PER_BATCH = int(os.getenv("MAX_TEXTS_PER_BATCH", "100"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "2000"))

app = FastAPI(title="Transdom Translation Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting: idetifies client by IP address and rejects requests
# beyond the configured limit with an HTTP 429 response.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

LANGUAGE_MODELS = {
    ("en", "pt"): {"model_name": "Helsinki-NLP/opus-mt-tc-big-en-pt", "target_tag": "por"},
    ("en", "es"): {"model_name": "Helsinki-NLP/opus-mt-en-es", "target_tag": None},
    ("en", "de"): {"model_name": "Helsinki-NLP/opus-mt-en-de", "target_tag": None},
}

# Where converted CTranslate2 models are cached on disk, so conversion
# only happens once per language pair, not on every server restart.
CT2_MODELS_DIR = "ct2_models"

MAX_LOADED_MODELS = 3
MAX_TRANSLATION_CACHE_SIZE = 5000

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
SIMILARITY_THRESHOLD = 0.92

loaded_models: OrderedDict = OrderedDict()
translation_cache: OrderedDict = OrderedDict()
semantic_cache: OrderedDict = OrderedDict()


def ensure_ct2_model(model_name: str, ct2_dir: str):
    if os.path.isdir(ct2_dir):
        return

    print(f"[transdom] Converting {model_name} to CTranslate2 (int8) — first run only")
    converter = TransformersConverter(model_name)
    converter.convert(ct2_dir, quantization="int8")


def get_model(source_lang: str, target_lang: str):
    pair_key = (source_lang, target_lang)

    if pair_key not in LANGUAGE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Language pair '{source_lang}-{target_lang}' is not supported.",
        )

    if pair_key in loaded_models:
        loaded_models.move_to_end(pair_key)
        return loaded_models[pair_key]

    model_name = LANGUAGE_MODELS[pair_key]["model_name"]
    ct2_dir = os.path.join(CT2_MODELS_DIR, f"{source_lang}-{target_lang}")

    ensure_ct2_model(model_name, ct2_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translator = ctranslate2.Translator(ct2_dir, compute_type="int8")
    loaded_models[pair_key] = (tokenizer, translator)

    if len(loaded_models) > MAX_LOADED_MODELS:
        oldest_key, _ = loaded_models.popitem(last=False)
        print(f"[transdom] Evicting model for {oldest_key} to free memory")

    return loaded_models[pair_key]


def find_similar_translation(source_lang: str, target_lang: str, embedding):
    best_match = None
    best_score = 0.0

    for (cached_lang_pair, _), entry in list(semantic_cache.items()):
        if cached_lang_pair != (source_lang, target_lang):
            continue

        score = util.cos_sim(embedding, entry["embedding"]).item()
        if score > best_score:
            best_score = score
            best_match = entry["translation"]

    if best_score >= SIMILARITY_THRESHOLD:
        return best_match
    return None


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    cache_key = (source_lang, target_lang, text)

    if cache_key in translation_cache:
        translation_cache.move_to_end(cache_key)
        return translation_cache[cache_key]

    embedding = embedding_model.encode(text, convert_to_tensor=True)
    similar_translation = find_similar_translation(source_lang, target_lang, embedding)
    if similar_translation is not None:
        translation_cache[cache_key] = similar_translation
        return similar_translation

    tokenizer, translator = get_model(source_lang, target_lang)
    target_tag = LANGUAGE_MODELS[(source_lang, target_lang)]["target_tag"]
    tagged_text = f">>{target_tag}<< {text}" if target_tag else text

    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(tagged_text))
    results = translator.translate_batch([source_tokens])
    target_tokens = results[0].hypotheses[0]
    translation = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens))

    translation_cache[cache_key] = translation
    if len(translation_cache) > MAX_TRANSLATION_CACHE_SIZE:
        translation_cache.popitem(last=False)

    semantic_key = ((source_lang, target_lang), text)
    semantic_cache[semantic_key] = {"embedding": embedding, "translation": translation}
    if len(semantic_cache) > MAX_TRANSLATION_CACHE_SIZE:
        semantic_cache.popitem(last=False)

    return translation

# Field(max_length=...) on plain str constrains character count.
# On a list, max_length constrauns the number of items. Combining both,
# via Annotated on the list's inner type, constrains each item AND the
# list size - rejecting oversized payloads before any code even runs.
BoundedText = Annotated[str, Field(max_length=MAX_TEXT_LENGTH)]

class TranslateRequest(BaseModel):
    text: BoundedText
    source_lang: str
    target_lang: str


class TranslateResponse(BaseModel):
    translation: str


class TranslateBatchRequest(BaseModel):
    texts: Annotated[list[BoundedText],  Field(max_length=MAX_TEXTS_PER_BATCH)]
    source_lang: str
    target_lang: str


class TranslateBatchResponse(BaseModel):
    translations: list[str]


@app.post("/translate", response_model=TranslateResponse)
@limiter.limit(RATE_LIMIT)
def translate(request: Request, body: TranslateRequest):
    translation = translate_text(body.text, body.source_lang, body.target_lang)
    return TranslateResponse(translation=translation)


@app.post("/translate/batch", response_model=TranslateBatchResponse)
@limiter.limit(RATE_LIMIT)
def translate_batch(request: Request, body: TranslateBatchRequest):
    translations = [
        translate_text(text, body.source_lang, body.target_lang)
        for text in body.texts
    ]
    return TranslateBatchResponse(translations=translations)