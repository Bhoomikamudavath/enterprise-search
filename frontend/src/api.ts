const API_BASE_URL = "http://127.0.0.1:8000";

export type SearchMode = "bm25" | "dense" | "hybrid";

export interface SearchResult {
  doc_id: string;
  title: string;
  snippet: string;
  score: number;
  url: string;
  tags: string[];
}

export interface SearchResponse {
  query: string;
  mode: string;
  results: SearchResult[];
}

export async function search(
  query: string,
  mode: SearchMode,
  topK: number = 10,
  tag?: string
): Promise<SearchResponse> {
  const params = new URLSearchParams({
    q: query,
    mode,
    top_k: String(topK),
  });
  if (tag) {
    params.set("tag", tag);
  }

  const response = await fetch(`${API_BASE_URL}/search?${params.toString()}`);

  if (!response.ok) {
    throw new Error(`Search failed: ${response.status} ${response.statusText}`);
  }

  return response.json();
}