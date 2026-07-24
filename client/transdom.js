// Transfom client library
// A class-based translation client that emits lifecycle events
// (start/success/error), so consuming code (vanilla JS, React, etc.)
// can react to what's happening — e.g. show a loading spinner.

// Tags whose text content should never be sent for translation
const IGNORED_TAGS = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "CODE", "PRE", "TEXTAREA"]);


class Transdom extends EventTarget {
  constructor(config = {}) {
    super();

    this.config = {
      apiUrl: "http://localhost:8000/translate/batch",
      sourceLang: "en",
      targetLang: "es",
      maxConsecutiveFailures: 3,
      ...config,    
    };

    // Instance-level state, not global - each Transdom instance tracks
    // its own translated nodes and its own MutationOserver.
    this. translatedNodes = new WeakSet();
    this.observe = null;
    this.mutationTimeout = null;

    // Circuit breaker state: trancks repeated failures so we can stop
    // hammerung a server that's clearly down, instead of retrying forever.
    this.consecutiveFailures = 0;
    this.circuitOpen = false;

    // Prevents overlapping translatePage() calls from running concurrently —
    // without this, rapid DOM mutations can fire multiple simultaneous
    // translation requests, each unaware of the others' failures.
    this.isTranslating = false;
  }

  collectTextNodes(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (node) => {
        const parentTag = node.parentElement ? node.parentElement.tagName : "";
        if (IGNORED_TAGS.has(parentTag)) return NodeFilter.FILTER_REJECT;
        if (!node.textContent.trim()) return NodeFilter.FILTER_REJECT;
        if (this.translatedNodes.has(node)) return NodeFilter.FILTER_REJECT;
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
  async translateTexts(texts) {
    const response = await fetch(this.config.apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        texts,
        source_lang: this.config.sourceLang,
        target_lang: this.config.targetLang,
      }),
    });

    if (!response.ok) {
      throw new Error(`Transdom server error: ${response.status}`);
    }

    const data = await response.json();
    return data.translations;
  } 

  // Main entry point: find all text on the page, translates it, and swaps it in.
  async translatePage() {
    if (this.circuitOpen || this.isTranslating) return;

    this.isTranslating = true;

    try {
      const textNodes = this.collectTextNodes(document.body);
      const originalTexts = textNodes.map((node) => node.textContent.trim());

      if (originalTexts.length === 0) return;

      this.dispatchEvent(new CustomEvent("translate:start", {
        detail: { count: originalTexts.length },
      }));

      try {
        const translations = await this.translateTexts(originalTexts);

        textNodes.forEach((node, index) => {
          node.textContent = translations[index];
          this.translatedNodes.add(node);
        });

        this.consecutiveFailures = 0;

        this.dispatchEvent(new CustomEvent("translate:success", {
          detail: { count: originalTexts.length },
        }));
      } catch (error) {
        this.consecutiveFailures += 1;

        this.dispatchEvent(new CustomEvent("translate:error", {
          detail: { error, consecutiveFailures: this.consecutiveFailures },
        }));

        if (this.consecutiveFailures >= this.config.maxConsecutiveFailures) {
          this.circuitOpen = true;
          this.stopAutoTranslate();

          this.dispatchEvent(new CustomEvent("translate:circuit-open", {
            detail: { consecutiveFailures: this.consecutiveFailures },
          }));
        }
      }
    } finally {
      // Always runs, whether translation succeeded, failed, or an
      // unexpected error was thrown — the lock must never get stuck "on".
      this.isTranslating = false;
    }
  }

  startAutoTranslate() {
    this.translatePage();

    this.observer = new MutationObserver(() => {
      clearTimeout(this.mutationTimeout);
      this.mutationTimeout = setTimeout(() => this.translatePage(), 300);
    });

    this.observer.observe(document.body, { childList: true, subtree: true });
  }

  stopAutoTranslate() {
    if (this.observer) {
      this.observer.disconnect();
      this.observer = null;
    }
    clearTimeout(this.mutationTimeout);
  }

  resetCircuit() {
    this.consecutiveFailures = 0;
    this.circuitOpen = false;
  }
}

window.Transdom = Transdom;
