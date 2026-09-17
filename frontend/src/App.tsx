import { useState } from "react";
import type { FormEvent } from "react";
import { search } from "./api";
import type { SearchMode, SearchResult } from "./api";
import "./App.css";

function escapeRegExp(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightSnippet(text: string, query: string) {
  const terms = query.trim().split(/\s+/).filter((t) => t.length > 1);
  if (terms.length === 0) return text;

  const pattern = new RegExp("(" + terms.map(escapeRegExp).join("|") + ")", "gi");
  const parts = text.split(pattern);

  return parts.map((part, i) => {
    const isMatch = terms.some((t) => t.toLowerCase() === part.toLowerCase());
    if (isMatch) {
      return <mark key={i}>{part}</mark>;
    }
    return part;
  });
}

const MODES: { value: SearchMode; label: string }[] = [
  { value: "bm25", label: "Keyword" },
  { value: "dense", label: "Semantic" },
  { value: "hybrid", label: "Hybrid" },
];

function App() {
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  async function runSearch(searchQuery: string, searchMode: SearchMode) {
    if (!searchQuery.trim()) {
      return;
    }

    setLoading(true);
    setError(null);
    setHasSearched(true);
    setSubmittedQuery(searchQuery);

    try {
      const response = await search(searchQuery, searchMode, 10);
      setResults(response.results);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("The search request failed.");
      }
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    runSearch(query, mode);
  }

  function handleModeChange(newMode: SearchMode) {
    setMode(newMode);
    if (submittedQuery) {
      runSearch(submittedQuery, newMode);
    }
  }

  return (
    <div className="app">
      <div className="topbar">
        <span className="app-name">Knowledge Search</span>
        <span className="app-desc">keyword plus semantic retrieval over engineering Q and A</span>
      </div>

      <form className="search-row" onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="error code, method name, or a question"
          className="search-input"
          autoFocus
        />
        <button type="submit" className="search-button" disabled={loading}>
          {loading ? "Searching" : "Search"}
        </button>
      </form>

      <div className="mode-tabs" role="tablist">
        {MODES.map((m) => {
          const isActive = mode === m.value;
          return (
            <button
              key={m.value}
              role="tab"
              aria-selected={isActive}
              className={isActive ? "mode-tab active" : "mode-tab"}
              onClick={() => handleModeChange(m.value)}
              type="button"
            >
              {m.label}
            </button>
          );
        })}
      </div>

      {error && (
        <div className="state-banner error">
          Search failed: {error}. Confirm the backend is running on port 8000.
        </div>
      )}

      {hasSearched && !loading && !error && results.length === 0 && (
        <div className="state-banner empty">
          No results for {submittedQuery}. Try fewer or more general terms.
        </div>
      )}

      {results.length > 0 && (
        <div className="result-count">
          {results.length} results for {submittedQuery}
        </div>
      )}

      <ul className="results">
        {results.map((result) => (
          <li key={result.doc_id} className="result-row">
            <a href={result.url} target="_blank" rel="noopener noreferrer" className="result-title">
              {result.title}
            </a>
            <div className="result-path">{result.url.replace("https://", "")}</div>
            <p className="result-snippet">{highlightSnippet(result.snippet, submittedQuery)}</p>
            <div className="result-meta">
              <span className="result-score">score {result.score}</span>
              {result.tags.slice(0, 4).map((tag) => (
                <span key={tag} className="result-tag">
                  {tag}
                </span>
              ))}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;