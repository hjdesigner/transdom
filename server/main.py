from collections import OrderedDict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

# Maximum number of models kept in memory at once, and maximum number
# of cached translations. Tune these based on how much RAM the machine
# running this server actually has.
MAX_LOADED_MODELS = 3
MAX_TRANSLATION_CACHE_SIZE = 5000

# OrderedDict lets us move an item to the end on access (marking it as
# "recently used") and pop the first item (the least recently used one)
# when we're over the limit.
loaded_models: OrderedDict = OrderedDict()
translation_cache: OrderedDict = OrderedDict()


def get_model(source_lang: str, target_lang: str):
    pair_key = (source_lang, target_lang)

    if pair_key not in LANGUAGE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Language pair '{source_lang}-{target_lang}' is not supported.",
        )

    if pair_key in loaded_models:
        # Accessed again — move it to the end so it's not seen as "old".
        loaded_models.move_to_end(pair_key)
        return loaded_models[pair_key]

    model_name = LANGUAGE_MODELS[pair_key]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    loaded_models[pair_key] = (tokenizer, model)

    if len(loaded_models) > MAX_LOADED_MODELS:
        # popitem(last=False) removes the FIRST item — the one that has
        # gone the longest without being used, since every access moves
        # items to the end.
        oldest_key, _ = loaded_models.popitem(last=False)
        print(f"[transdom] Evicting model for {oldest_key} to free memory")

    return loaded_models[pair_key]


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    cache_key = (source_lang, target_lang, text)

    if cache_key in translation_cache:
        translation_cache.move_to_end(cache_key)
        return translation_cache[cache_key]

    tokenizer, model = get_model(source_lang, target_lang)
    target_tag = LANGUAGE_MODELS[(source_lang, target_lang)]["target_tag"]
    tagged_text = f">>{target_tag}<< {text}" if target_tag else text

    inputs = tokenizer(tagged_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    translation_cache[cache_key] = translation
    if len(translation_cache) > MAX_TRANSLATION_CACHE_SIZE:
        translation_cache.popitem(last=False)

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