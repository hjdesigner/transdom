# transdom

Real-time translation for any web page — powered by your own self-hosted
[Transdom server](https://github.com/hjdesigner/transdom).

Transdom scans a page's DOM for text, sends it to your translation server,
and swaps the translated text back in — including content added later
(e.g. by a framework re-rendering), via `MutationObserver`.

## Installation

```bash
npm install transdom
```

## Usage (vanilla JS)

```js
import { Transdom } from "transdom";

const transdom = new Transdom({
  apiUrl: "http://your-server:8000/translate/batch",
  sourceLang: "en",
  targetLang: "pt",
});

transdom.addEventListener("translate:start", () => console.log("Translating..."));
transdom.addEventListener("translate:success", () => console.log("Done!"));
transdom.addEventListener("translate:error", (e) => console.error(e.detail.error));

transdom.startAutoTranslate();
```

## Usage (React)

```jsx
import { useRef } from "react";
import { useTransdom } from "transdom/react";

function App() {
  const contentRef = useRef(null);
  const { status, error, translate, stop } = useTransdom(
    { apiUrl: "http://your-server:8000/translate/batch", targetLang: "pt" },
    contentRef
  );

  return (
    <div>
      <div ref={contentRef}>
        <h1>Welcome to our website</h1>
      </div>
      <button onClick={translate}>Translate</button>
    </div>
  );
}
```

**Note:** wrap only the translatable content in the `ref` passed to
`useTransdom` — keep loading/error UI outside of it, otherwise Transdom may
detect its own status messages as new content to translate.

## Configuration options

| Option | Default | Description |
|---|---|---|
| `apiUrl` | `http://localhost:8000/translate/batch` | Your Transdom server's batch endpoint |
| `sourceLang` | `"en"` | Source language code |
| `targetLang` | `"pt"` | Target language code |
| `root` | `document.body` | Element to scan/watch for translatable text |
| `maxConsecutiveFailures` | `3` | Failures before the circuit breaker stops retrying |

## Requires a Transdom server

This package is the client half only. You need a running
[Transdom server](https://github.com/hjdesigner/transdom) (self-hosted,
via Docker) to actually translate text.

## License

MIT