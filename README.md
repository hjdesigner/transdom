# Transdom

Real-time translation for any web page — self-hosted, powered by open-source AI models.

Transdom has two pieces:
- A **translation server** (Python + FastAPI) you run yourself — like Strapi, you host it, you own the data and the cost.
- A **client library** (`transdom.js`) you drop into any web page to translate its content in real time.

## Contents

- [Quick start](#quick-start)
- [Running tests](#running-tests)
- [Supported languages](#supported-languages)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Glossary](#glossary)
- [Memory management](#memory-management)
- [Semantic caching](#semantic-caching)
- [Translation engine](#translation-engine)
- [Deployment & resource requirements](#deployment--resource-requirements)
- [Security considerations](#security-considerations)
- [License](#license)

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

The server from step 1 must be running — the client library has nothing to translate without it.

```bash
npm install transdom
```

```js
import { Transdom } from "transdom";

const transdom = new Transdom({
  apiUrl: "http://your-server:8000/translate/batch",
  sourceLang: "en",
  targetLang: "pt",
});

transdom.startAutoTranslate();
```

**For React:**

```jsx
import { useRef } from "react";
import { useTransdom } from "transdom/react";

function App() {
  const contentRef = useRef(null);

  const { status, error, translate, stop } = useTransdom(
    {
      apiUrl: "http://your-server:8000/translate/batch",
      sourceLang: "en",
      targetLang: "pt",
    },
    contentRef
  );

  return (
    <div>
      {/* Only this block is scanned/watched by Transdom */}
      <div ref={contentRef}>
        <h1>Welcome to our website</h1>
        <p>This is a simple paragraph used to test automatic translation.</p>
      </div>

      {/* Everything below lives OUTSIDE the translatable area,
          so Transdom never sees (or reacts to) its own status UI */}
      <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
        <button onClick={translate} disabled={status === "loading"}>
          {status === "loading" ? "Translating..." : "Translate Page"}
        </button>
        <button onClick={stop}>Stop</button>
      </div>

      {status === "error" && <p style={{ color: "red" }}>Error: {error?.message}</p>}
      {status === "success" && <p style={{ color: "green" }}>Translated!</p>}
    </div>
  );
}
```

## Running tests

```bash
cd server
pip install pytest httpx
python -m pytest test_api.py -v
```

Tests mock the translation model and tokenizer (no downloads, no GPU needed)
and cover glossary matching, LRU eviction, translation caching, and request validation.

## Supported languages

| Source | Target |
|--------|--------|
| en     | pt     |
| en     | es     |
| en     | de     |

### Adding a new language pair

1. Find the model for your language pair on the [Helsinki-NLP OPUS-MT models page](https://huggingface.co/Helsinki-NLP). Search for `opus-mt-{source}-{target}` (e.g. `opus-mt-en-ja` for English → Japanese). Not every pair exists — check the model actually loads before relying on it.
2. Open the model's page and check its README for a "language codes" or "valid target labels" section. Some models (usually named `tc-big` or covering multiple related languages) require a `>>xxx<<` tag prefix to pick the exact target — like `Helsinki-NLP/opus-mt-tc-big-en-pt` does with `>>por<<`/`>>pob<<`. Most bilingual models (`opus-mt-en-es`, for example) don't need one — use `null`/`None` in that case.
3. Add an entry to `LANGUAGE_MODELS` in `server/main.py`:

```python
("en", "ja"): {"model_name": "Helsinki-NLP/opus-mt-en-ja", "target_tag": None},
```

4. Restart the server. The model downloads and converts to CTranslate2 automatically the first time that pair is used — no other setup needed.

Contributions adding new verified language pairs are welcome via pull request.

## How it works

1. `transdom.js` scans the page's DOM for text nodes (skipping `<script>`, `<style>`, etc.) and watches for new content added later via `MutationObserver`.
2. It sends batches of text to the server's `/translate/batch` endpoint.
3. The server runs the appropriate open-source translation model (from Hugging Face) and returns the translated text.
4. The client swaps the translated text back into the page in place, without touching HTML structure or event listeners.

## Configuration

Copy `.env.example` to `.env` (inside `server/`) and adjust as needed:

- **ALLOWED_ORIGINS** — comma-separated list of domains allowed to call this API from a browser (CORS). Restrict this to your own site's domain(s) in production; the default only allows `localhost:5500`, used by the included test page.
- **RATE_LIMIT** — max requests per IP address (e.g. `30/minute`, `100/hour`), enforced with [slowapi](https://github.com/laurentS/slowapi).
- **MAX_TEXTS_PER_BATCH** / **MAX_TEXT_LENGTH** — reject oversized requests (too many texts, or texts that are too long) before they reach the translation engine, protecting the server from abuse or accidental misuse.
- **HF_TOKEN** — optional Hugging Face access token (read-only is enough). Not required, but avoids hitting anonymous rate limits when downloading translation models. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

If no `.env` file is present, the server falls back to safe defaults, so it still runs out of the box for local testing.

## Glossary

Create a `glossary.json` file inside `server/` to override automatic translation for specific terms — useful for brand names, technical terms, or UI strings where you want an exact, consistent translation instead of whatever the AI model generates.

```json
{
  "do_not_translate": ["Transdom", "GitHub", "iPhone"],
  "custom_translations": {
    "en-pt": {
      "Login": "Entrar",
      "Sign up": "Criar conta"
    }
  }
}
```

- **do_not_translate** — terms (case-sensitive, whole-word match) that are never translated, whether they're the entire text or appear inside a larger sentence.
- **custom_translations** — per language-pair overrides for specific terms, applied the same way: whole text or inside a sentence.

Glossary rules take priority over caching and the AI model. When a term appears inside a larger sentence, it's temporarily replaced with a name-like placeholder before translation (name-like tokens survive translation far more reliably than symbols or raw variable names) and restored afterward. If `glossary.json` doesn't exist, the server runs normally with no glossary rules.

## Memory management

Translation models are loaded into RAM on first use per language pair, and cached in-memory results avoid re-translating the same text twice. Both caches are bounded using an **LRU (Least Recently Used) eviction policy** — once a limit is reached, the least recently used entry is dropped to make room for new ones, so memory usage stays predictable instead of growing forever.

These limits are configurable in `server/main.py`:

```python
MAX_LOADED_MODELS = 3              # max translation models kept in RAM at once
MAX_TRANSLATION_CACHE_SIZE = 5000  # max cached translated strings
```

Tune `MAX_LOADED_MODELS` based on available RAM — each model uses roughly 200–400MB after CTranslate2/int8 quantization.

## Semantic caching

Beyond exact-match caching, Transdom also caches by **meaning**. Each translated text is converted into a vector embedding (using `all-MiniLM-L6-v2`, from the `sentence-transformers` library), and future requests are compared against cached embeddings using cosine similarity. If a new text is semantically close enough to something already translated — even with different wording or word order — the cached translation is reused instead of running the translation model again.

Example: `"You have successfully logged in"` and `"You have logged in successfully"` are different strings, but nearly identical in meaning (similarity score ≈ 0.99). Semantic caching catches this; exact-match caching would not.

The similarity threshold is configurable in `server/main.py`:

```python
SIMILARITY_THRESHOLD = 0.92  # 0 to 1 — how close two texts must be in meaning
                              # to reuse a cached translation
```

Lower values reuse more aggressively (faster, but risk merging texts that don't actually mean the same thing). Higher values are safer but closer to exact-match caching. `0.92` was chosen by testing real examples rather than picked arbitrarily — tune it based on the kind of text your site uses.

This feature can be disabled with `ENABLE_SEMANTIC_CACHE=false` in `.env` if you'd rather skip it. **Note:** this was measured to save only ~6% of baseline memory usage (see below) — the bulk of the server's baseline comes from elsewhere (CTranslate2 and transformers overhead), not from this feature. Disabling it is not an effective way to reduce memory footprint; it exists purely as an optional trade-off between cache quality and a small amount of RAM.

## Translation engine

Translation models run on **CTranslate2** with **int8 quantization**, instead of plain PyTorch. This was measured (not assumed) to give a ~6x speedup and roughly halve the model's disk/memory footprint, with no observable quality difference on test sentences. Converted models are cached in `ct2_models/` and generated automatically on first use per language pair — no manual setup required.

## Deployment & resource requirements

Transdom runs anywhere Docker does — there's no dependency on a specific hosting provider. On any VPS or cloud platform with Docker installed:

```bash
git clone https://github.com/hjdesigner/transdom
cd transdom
docker compose up --build -d
```

**Measured RAM usage** (via `docker stats`, one enabled language pair):

| State | RAM usage |
|---|---|
| Server idle (no translation yet) | ~378 MB |
| After one translation (model loaded) | ~710 MB |

This baseline comes from Python + CTranslate2 + transformers overhead — a mostly fixed cost regardless of how many language pairs are enabled. As a result, hosting tiers below ~1GB of RAM (most providers' free tiers) are not viable. Budget for at least a 1–2GB instance in production.

## Security considerations

Transdom includes some protections out of the box, but self-hosting means
you're responsible for the rest. Being upfront about both:

**Covered:**
- **Per-IP rate limiting** (`RATE_LIMIT` in `.env`) — throttles a single IP sending too many requests too fast.
- **Payload limits** (`MAX_TEXT_LENGTH`, `MAX_TEXTS_PER_BATCH`) — rejects oversized requests before they reach the translation engine.
- **CORS** (`ALLOWED_ORIGINS`) — restricts which domains can call the API *from a browser*.

**Not covered — these are the operator's responsibility:**
- **Distributed abuse.** Per-IP rate limiting doesn't stop many different IPs hitting the server at once. If this matters for your deployment, put a proxy/CDN (e.g. Cloudflare, even on a free plan) in front of the server.
- **CORS is not an access control.** It only restricts browser-based JavaScript. A direct HTTP request (`curl`, a script, another server) ignores CORS entirely and reaches the API the same as a browser would.
- **Rate limiting counts requests, not cost.** A client stays "within limits" while still sending the maximum allowed batch size on every request — each of which runs real AI inference. The current limits don't distinguish cheap requests from expensive ones.
- **No authentication.** Anyone who knows the server's URL can use it. Fine for a personal site calling its own server; add an API key or auth layer if that's not your case.
- **No HTTPS built in.** The server speaks plain HTTP. Put it behind a reverse proxy (Nginx, Caddy, Traefik) handling TLS before exposing it publicly.

None of this is unusual for a self-hosted tool — it's the same shape of trade-off as running any other open-source server yourself. The goal here is to be explicit about where the line sits, rather than leave it implicit.

## License

MIT — see [LICENSE](LICENSE) for details.