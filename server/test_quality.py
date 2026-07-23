import ctranslate2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu.metrics import BLEU, CHRF

MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-pt"
CT2_MODEL_DIR = "ct2_models/en-pt"

# A small, hand-picked test set: source text + a trusted human reference
# translation. This is the "answer key" the metrics compare against.
test_cases = [
    {
        "source": "Welcome to our website",
        "reference": "Bem-vindo ao nosso site",
    },
    {
        "source": "Please enter your email address",
        "reference": "Por favor, insira o seu endereço de e-mail",
    },
    {
        "source": "Your order has been shipped and will arrive soon",
        "reference": "O seu pedido foi enviado e chegará em breve",
    },
    {
        "source": "You have successfully logged in",
        "reference": "Você entrou com sucesso",
    },
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def translate_with_pytorch(text: str) -> str:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    inputs = tokenizer(f">>por<< {text}", return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def translate_with_ctranslate2(text: str) -> str:
    translator = ctranslate2.Translator(CT2_MODEL_DIR, compute_type="int8")
    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(f">>por<< {text}"))
    results = translator.translate_batch([source_tokens])
    target_tokens = results[0].hypotheses[0]
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens))


def evaluate(translate_fn, engine_name: str):
    bleu = BLEU()
    chrf = CHRF()

    candidates = []
    references = [[]]  # sacrebleu expects a list of reference sets

    print(f"\n--- {engine_name} ---")
    for case in test_cases:
        candidate = translate_fn(case["source"])
        candidates.append(candidate)
        references[0].append(case["reference"])

        sentence_bleu = bleu.sentence_score(candidate, [case["reference"]])
        print(f"  {case['source']!r}")
        print(f"    candidate:  {candidate}")
        print(f"    reference:  {case['reference']}")
        print(f"    BLEU: {sentence_bleu.score:.1f}")

    corpus_bleu = bleu.corpus_score(candidates, references)
    corpus_chrf = chrf.corpus_score(candidates, references)
    print(f"\n  Overall BLEU: {corpus_bleu.score:.1f}")
    print(f"  Overall chrF: {corpus_chrf.score:.1f}")


evaluate(translate_with_pytorch, "PyTorch (float32)")
evaluate(translate_with_ctranslate2, "CTranslate2 (int8)")