// Transfom client library
// Scans the current page for translatable text, sends it to a Transdom
// server, and replaces the text in place - without touching HTML structure
// or breaking event listeners attached to elements.

const TRANSDOM_CONFIG = {
  apiUrl: "http://localhost:8000/translate/batch",
  sourceLang: "en",
  targetLang: "pt",
}

// Tags whose text content should never be sent for translation
const IGNORE_TAGS = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "CODE", "PRE", "TEXTAREA"]);

// Tracks witch text node have aldeady been translated, so we never
// send the same node the server twice - and so a re-scan triggered
// by MutationObserver only priocesses genuinely new content.
const translatedNodes = new WeakSet();

function collectTextNodes(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parentTag = node.parentElement ? node.parentElement.tagName : "";
      if (IGNORE_TAGS.has(parentTag)) return NodeFilter.FILTER_REJECT;
      if (!node.textContent.trim()) return NodeFilter.FILTER_REJECT;
      if (translatedNodes.has(node)) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
     },
  });

  const nodes = [];
  let current = walker.nextNode();
  while (current) {
    nodes.push(current);
    current = walker.nextNode();
  }
  return nodes;
}

// Calls the Transdom server's batch endpoint and returns the translations.
// in the same order as the texts sent in
async function translateTexts(texts, sourceLang, targetLang) {
  const response = await fetch(TRANSDOM_CONFIG.apiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ texts, source_lang: sourceLang, target_lang: targetLang }),
  });

  if (!response.ok) {
    throw new Error(`Transdom server error: ${response.status}`);
  }

  const data = await response.json();
  return data.translations;
}


// Main entry point: find all text on the page, translates it, and swaps it in.
async function translatePage() {
  const textNodes = collectTextNodes(document.body);
  const oridinalTexts = textNodes.map(node => node.textContent.trim());

  if (oridinalTexts.length === 0) return;

  const translations = await translateTexts(
    oridinalTexts,
    TRANSDOM_CONFIG.sourceLang,
    TRANSDOM_CONFIG.targetLang
  );

  textNodes.forEach((node, index) => {
    node.textContent = translations[index];
    translatedNodes.add(node);
  });
}

let mutationTimeout = null;

// Starts translating the page, then keeps watching for new content
// added later (e.g. by framework re-rendering, or content loaded)
// asynchronously and translates it automatically.
function startAutoTranslate() {
  translatePage();

  const observer = new MutationObserver(() => {
    // A single DOM update can trigger many mutation events at once
    // (e.g. a whole component re-rendering adds several nodes).
    // we wait a bit and react only onde, instead of per mutation.
    clearTimeout(mutationTimeout);
    mutationTimeout = setTimeout(translatePage, 300);
  });

  observer.observe(document.body, { childList: true, subtree: true });

  return observer;
}


window.Transdom = { translatePage, startAutoTranslate, config: TRANSDOM_CONFIG };