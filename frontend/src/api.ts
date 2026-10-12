export type Named = { name: string; description: string }

export type QueryScore = {
  query_id: string
  recall: number
  rr: number
  ndcg: number
  latency_ms: number
  top_ids: string[]
}

export type RetrieverReport = {
  name: string
  recall: number
  mrr: number
  ndcg: number
  p50_ms: number
  p95_ms: number
  per_query: QueryScore[]
}

export type Evaluation = {
  dataset: string
  k: number
  queries: Record<string, { text: string; relevant: string[] }>
  retrievers: RetrieverReport[]
}

export type Hit = { doc_id: string; score: number; title: string; snippet: string; relevant: boolean | null }
export type SearchResponse = {
  query: string
  query_id: string | null
  relevant: string[]
  columns: { retriever: string; latency_ms: number; results: Hit[] }[]
}
export type Doc = { title: string; snippet: string }

async function call<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `Request failed (${res.status})`)
  return res.json()
}

export const api = {
  datasets: () => call<Named[]>('/datasets'),
  retrievers: () => call<Named[]>('/retrievers'),
  evaluate: (dataset: string, retrievers: string[], k: number) =>
    call<Evaluation>('/evaluate', { dataset, retrievers, k }),
  search: (query: string, dataset: string, retrievers: string[], k: number, query_id?: string) =>
    call<SearchResponse>('/search', { query, dataset, retrievers, k, query_id }),
  docs: (dataset: string, ids: string[]) =>
    call<Record<string, Doc>>(`/datasets/${dataset}/docs?ids=${encodeURIComponent(ids.join(','))}`),
}

export const LABELS: Record<string, string> = { bm25: 'BM25', dense: 'Dense', hybrid: 'Hybrid' }
export const label = (name: string) => LABELS[name] ?? name
