from fastapi import FastAPI, HTTPException, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(title="Transdom Translation Server")

# Allow any website to call this API from the browser.
# In a production self-hosted setuo, you'd usually restrict this
# to especific domains instead of "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registry of supported language pairs.
# "target_tag" is only need for multilingual models like en-pt,
# which require a >>xxx<< prefix to pick the target variant.

LANGUAGE_MODELS = {
    ("en", "pt"): {"model_name": "Helsinki-NLP/opus-mt-tc-big-en-pt", "target_tag": "por"},
    ("en", "es"): {"model_name": "Helsinki-NLP/opus-mt-en-es", "target_tag": None},
    ("en", "de"): {"model_name": "Helsinki-NLP/opus-mt-en-de", "target_tag": None},
}

# In-memory cache: keeps already-loaded models so we don't reload them
# on every request. Key = (source_lang, target_lang), valure = (tokenizer, model  )
loaded_models = dict = {}

# translation cache: key = (source_lang, target_lang, text), value = translation text.
# this avoids re-running the model for text we've already translated before.
translation_cache = dict = {}

def get_model(source_lang: str, target_lang: str):
    pair_key = (source_lang, target_lang)

    if pair_key not in LANGUAGE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Language pair '{source_lang}-{target_lang}' is not supported.",
        )
    
    if pair_key not in loaded_models:
        model_name = LANGUAGE_MODELS[pair_key]["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        loaded_models[pair_key] = (tokenizer, model)

    return loaded_models[pair_key]

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    cache_key = (source_lang, target_lang, text)

    if cache_key in translation_cache:
        return translation_cache[cache_key]
    
    tokenizer, model = get_model(source_lang, target_lang)
    target_tag = LANGUAGE_MODELS[(source_lang, target_lang)]["target_tag"]
    tagged_text = f">>{target_tag}<< {text}" if target_tag else text

    inputs = tokenizer(tagged_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    translation_cache[cache_key] = translation
    return translation


class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

class TranslateResponse(BaseModel):
    translation: str


@app.post("/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest):
    translation = translate_text(request.text, request.source_lang, request.target_lang)
    return TranslateResponse(translation=translation)


class TranslateBatchRequest(BaseModel):
    texts: list[str]
    source_lang: str
    target_lang: str

class TranslateBatchResponse(BaseModel):
    translations: list[str]
    

@app.post("/translate/batch", response_model=TranslateBatchResponse)
def translate_batch(request: TranslateBatchRequest):
    translations = [
        translate_text(text, request.source_lang, request.target_lang)
        for text in request.texts
    ]
    return TranslateBatchResponse(translations=translations)