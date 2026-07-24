import { useRef } from "react";
import { useTransdom } from "./useTransdom";

function App() {
  const contentRef = useRef(null);

  const { status, error, translate, stop } = useTransdom(
    {
      apiUrl: "http://localhost:8000/translate/batch",
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

      <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 16 }}>
        {/* Everything below lives OUTSIDE the translatable area,
            so Transdom never sees (or reacts to) its own status UI */}
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

export default App;