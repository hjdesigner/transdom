import os
from collections import OrderedDict
import ctranslate2
from ctranslate2.converters import TransformersConverter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

app = FastAPI(title="Transdom Translation Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str


class TranslateResponse(BaseModel):
    translation: str


class TranslateBatchRequest(BaseModel):
    texts: list[str]
    source_lang: str
    target_lang: str


class TranslateBatchResponse(BaseModel):
    translations: list[str]


@app.post("/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest):
    translation = translate_text(request.text, request.source_lang, request.target_lang)
    return TranslateResponse(translation=translation)


@app.post("/translate/batch", response_model=TranslateBatchResponse)
def translate_batch(request: TranslateBatchRequest):
    translations = [
        translate_text(text, request.source_lang, request.target_lang)
        for text in request.texts
    ]
    return TranslateBatchResponse(translations=translations)