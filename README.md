# Transdom

Real-time translation for any web page — self-hosted, powered by open-source AI models.

Transdom has two pieces:
- A **translation server** (Python + FastAPI) you run yourself — like Strapi, you host it, you own the data and the cost.
- A **client library** (`transdom.js`) you drop into any web page to translate its content in real time.

## Quick start

### 1. Run the server

```bash
docker compose up --build
```

The server will be available at `http://localhost:8000`.

### 2. Add the client to your page

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

## License

MIT — see [LICENSE](LICENSE) for details.