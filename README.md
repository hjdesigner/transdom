# Transdom

Real-time translation for any web page — self-hosted, powered by open-source AI models.

Transdom has two pieces:
- A **translation server** (Python + FastAPI) you run yourself — like Strapi, you host it, you own the data and the cost.
- A **client library** (`transdom.js`) you drop into any web page to translate its content in real time.

## Quick start

### 1. Run the server

**Option A — Docker (recommended):**

```bash
docker compose up --build
```

**Option B — Local Python:**

```bash
cd server
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Either way, the server will be available at `http://localhost:8000`. Interactive docs (test it without writing any code) live at `http://localhost:8000/docs`.

### 2. Serve the client library and try it

```bash
cd client
python -m http.server 5500
```

Open `http://localhost:5500/test.html` in your browser.

### 3. Add Transdom to your own page

```html
<script src="transdom.js"></script>
<script>
  Transdom.startAutoTranslate();
</script>
```

## Supported languages

| Source | Target |
|--------|--------|
| en     | pt     |
| en     | es     |
| en     | de     |

Add more pairs by editing `LANGUAGE_MODELS` in `server/main.py`.

## How it works

1. `transdom.js` scans the page's DOM for text nodes (skipping `<script>`, `<style>`, etc.) and watches for new content added later via `MutationObserver`.
2. It sends batches of text to the server's `/translate/batch` endpoint.
3. The server runs the appropriate open-source translation model (from Hugging Face) and returns the translated text.
4. The client swaps the translated text back into the page in place, without touching HTML structure or event listeners.

## Memory management

Translation models are loaded into RAM on first use per language pair, and cached in-memory results avoid re-translating the same text twice. Both caches are bounded using an **LRU (Least Recently Used) eviction policy** — once a limit is reached, the least recently used entry is dropped to make room for new ones, so memory usage stays predictable instead of growing forever.

These limits are configurable in `server/main.py`:

```python
MAX_LOADED_MODELS = 3           # max translation models kept in RAM at once
MAX_TRANSLATION_CACHE_SIZE = 5000  # max cached translated strings
```

Tune `MAX_LOADED_MODELS` based on available RAM — each model uses roughly 300MB–2GB depending on the language pair.

## Semantic caching

Beyond exact-match caching, Transdom also caches by **meaning**. Each translated
text is converted into a vector embedding (using `all-MiniLM-L6-v2`, from the
`sentence-transformers` library), and future requests are compared against
cached embeddings using cosine similarity. If a new text is semantically close
enough to something already translated — even with different wording or word
order — the cached translation is reused instead of running the translation
model again.

Example: `"You have successfully logged in"` and `"You have logged in
successfully"` are different strings, but nearly identical in meaning
(similarity score ≈ 0.99). Semantic caching catches this; exact-match caching
would not.

The similarity threshold is configurable in `server/main.py`:

```python
SIMILARITY_THRESHOLD = 0.92  # 0 to 1 — how close two texts must be in meaning
                              # to reuse a cached translation
```

Lower values reuse more aggressively (faster, but risk merging texts that
don't actually mean the same thing). Higher values are safer but closer to
exact-match caching. `0.92` was chosen by testing real examples rather than
picked arbitrarily — tune it based on the kind of text your site uses.

## Translation engine

Translation models run on **CTranslate2** with **int8 quantization**, instead
of plain PyTorch. This was measured (not assumed) to give a ~6x speedup and
roughly halve the model's disk/memory footprint, with no observable quality
difference on test sentences. Converted models are cached in `ct2_models/`
and generated automatically on first use per language pair — no manual setup
required.

## License

MIT — see [LICENSE](LICENSE) for details.