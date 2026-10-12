import { useEffect, useState } from 'react'
import { api, label, type SearchResponse } from '../api'
import { ms } from '../format'

type DatasetQuery = { id: string; text: string; relevant: string[] }

/** The query and dataset-query id saved in the URL, e.g. #/search?q=vitamin+d&id=3 */
function linkedQuery() {
  const params = new URLSearchParams(window.location.hash.split('?')[1] ?? '')
  return { q: params.get('q') ?? '', id: params.get('id') ?? undefined }
}

function rememberQuery(text: string, queryId?: string | null) {
  const params = new URLSearchParams({ q: text })
  if (queryId) params.set('id', queryId)
  // replaceState updates the address bar without adding a history entry or re-rendering the app.
  window.history.replaceState(null, '', `#/search?${params}`)
}

export default function LiveSearch({ dataset, k, retrievers }: { dataset: string; k: number; retrievers: string[] }) {
  const [query, setQuery] = useState(() => linkedQuery().q)
  const [copied, setCopied] = useState(false)
  const [samples, setSamples] = useState<DatasetQuery[]>([])
  const [result, setResult] = useState<SearchResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`/api/datasets/${dataset}/queries`)
      .then((r) => r.json())
      .then(setSamples)
      .catch(() => setSamples([]))
  }, [dataset])

  async function run(text: string, queryId?: string) {
    if (!text.trim() || !retrievers.length) return
    setBusy(true)
    setError('')
    try {
      const response = await api.search(text, dataset, retrievers, k, queryId)
      setResult(response)
      rememberQuery(text, response.query_id)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  // Opening a shared link runs its query right away.
  useEffect(() => {
    const { q, id } = linkedQuery()
    if (q && retrievers.length) run(q, id)
    // run only once the retrievers are known; later searches come from the form
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [retrievers.length])

  async function copyLink() {
    await navigator.clipboard.writeText(window.location.href)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  function randomQuery() {
    if (!samples.length) return
    const q = samples[Math.floor(Math.random() * samples.length)]
    setQuery(q.text)
    run(q.text, q.id)
  }

  return (
    <>
      <h1>Live search</h1>
      <p className="lede">
        Type anything and compare each retriever's results side by side. Pick a query from the dataset to see which
        results a human marked relevant.
      </p>
      <form
        className="searchbar"
        onSubmit={(e) => {
          e.preventDefault()
          run(query)
        }}
      >
        <input
          className="input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. vitamin D deficiency increases the risk of depression"
          autoFocus
        />
        <button className="btn primary" disabled={busy}>
          {busy ? 'Searching…' : 'Search'}
        </button>
        <button type="button" className="btn" onClick={randomQuery} disabled={!samples.length || busy}>
          Random {dataset} query
        </button>
      </form>

      {error && <div className="status error">{error}</div>}
      {result && (
        <>
          <p className="query-note">
            {result.query_id
              ? `Dataset query ${result.query_id}: ${result.relevant.length} relevant document${result.relevant.length === 1 ? '' : 's'} highlighted.`
              : 'Custom query: there are no relevance judgments, so nothing is highlighted.'}{' '}
            <button type="button" className="linkish" onClick={copyLink}>
              {copied ? 'Link copied' : 'Copy link to these results'}
            </button>
          </p>
          <div className="columns" style={{ ['--n' as string]: result.columns.length }}>
            {result.columns.map((col) => (
              <div key={col.retriever}>
                <div className="col-head">
                  {label(col.retriever)}
                  <span className="mono">{ms(col.latency_ms)} ms</span>
                </div>
                {col.results.map((h, i) => (
                  <div key={h.doc_id} className={`hit ${h.relevant ? 'rel' : ''}`}>
                    <span className="rank mono">{i + 1}</span>
                    <span className="title">{h.title || h.doc_id}</span>
                    <span className="snippet">{h.snippet}</span>
                    <span className="meta mono">
                      {h.relevant && <span className="tag">RELEVANT</span>}
                      <span>score {h.score.toFixed(3)}</span>
                      <span>#{h.doc_id}</span>
                    </span>
                  </div>
                ))}
                {!col.results.length && <div className="muted">No results.</div>}
              </div>
            ))}
          </div>
        </>
      )}
      {!result && !error && (
        <div className="panel">
          <div className="status">Results from each retriever will appear here in side-by-side columns.</div>
        </div>
      )}
    </>
  )
}
