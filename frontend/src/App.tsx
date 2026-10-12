import { useEffect, useState } from 'react'
import { api, label, type Evaluation, type Named } from './api'
import Leaderboard from './pages/Leaderboard'
import LiveSearch from './pages/LiveSearch'
import QueryExplorer from './pages/QueryExplorer'

const PAGES = [
  { id: 'leaderboard', label: 'Leaderboard' },
  { id: 'explorer', label: 'Query explorer' },
  { id: 'search', label: 'Live search' },
] as const
type Page = (typeof PAGES)[number]['id']

function currentPage(): Page {
  // The hash looks like #/search?q=..., so the page id is the part before any query string.
  const id = window.location.hash.replace('#/', '').split('?')[0]
  return PAGES.some((p) => p.id === id) ? (id as Page) : 'leaderboard'
}

export default function App() {
  const [page, setPage] = useState<Page>(currentPage)
  const [datasets, setDatasets] = useState<Named[]>([])
  const [retrievers, setRetrievers] = useState<Named[]>([])
  const [dataset, setDataset] = useState('scifact')
  const [k, setK] = useState(10)
  const [hidden, setHidden] = useState<string[]>([]) // retrievers switched off in the header
  // Results are stored with the dataset and k they belong to, so switching either shows the loading state.
  const [loaded, setLoaded] = useState<{ key: string; evaluation: Evaluation | null; error: string } | null>(null)

  useEffect(() => {
    const onHash = () => setPage(currentPage())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    api.datasets().then(setDatasets).catch(() => {})
    api.retrievers().then(setRetrievers).catch(() => {})
  }, [])

  // The leaderboard and the query explorer share one evaluation per dataset and k.
  useEffect(() => {
    if (!retrievers.length) return
    let stale = false
    const key = `${dataset}|${k}`
    api
      .evaluate(dataset, retrievers.map((r) => r.name), k)
      .then((evaluation) => !stale && setLoaded({ key, evaluation, error: '' }))
      .catch((e: Error) => !stale && setLoaded({ key, evaluation: null, error: e.message }))
    return () => {
      stale = true
    }
  }, [dataset, k, retrievers])

  const current = loaded?.key === `${dataset}|${k}` ? loaded : null
  const names = retrievers.map((r) => r.name).filter((n) => !hidden.includes(n))
  // Every retriever is evaluated once; switching one off only filters what the pages show.
  const evaluation = current?.evaluation
    ? { ...current.evaluation, retrievers: current.evaluation.retrievers.filter((r) => names.includes(r.name)) }
    : null
  const evalError = current?.error ?? ''
  const props = { dataset, k, evaluation, evalError, retrievers }
  const toggle = (name: string) =>
    setHidden((h) => (h.includes(name) ? h.filter((n) => n !== name) : names.length > 1 ? [...h, name] : h))

  return (
    <>
      <header className="top">
        <div className="brand">
          <svg width="20" height="20" viewBox="0 0 32 32" aria-hidden>
            <path d="M6 19 16 9l10 10-4 0-6-6-6 6Z" fill="currentColor" />
            <circle cx="16" cy="23" r="2" fill="currentColor" />
          </svg>
          kestrel <small>retrieval eval</small>
        </div>
        <nav className="tabs">
          {PAGES.map((p) => (
            <a key={p.id} href={`#/${p.id}`} className={`tab ${page === p.id ? 'on' : ''}`}>
              {p.label}
            </a>
          ))}
        </nav>
        <div className="controls">
          <div className="pills" role="group" aria-label="Retrievers shown">
            {retrievers.map((r) => (
              <button
                key={r.name}
                className={`pill ${hidden.includes(r.name) ? '' : 'on'}`}
                onClick={() => toggle(r.name)}
                title={r.description}
                aria-pressed={!hidden.includes(r.name)}
              >
                {label(r.name)}
              </button>
            ))}
          </div>
          <label className="control">
            dataset
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {(datasets.length ? datasets : [{ name: 'scifact', description: '' }]).map((d) => (
                <option key={d.name} value={d.name} title={d.description}>
                  {d.name}
                </option>
              ))}
            </select>
          </label>
          <label className="control">
            k
            <select value={k} onChange={(e) => setK(Number(e.target.value))}>
              {[5, 10, 20, 50].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>
      <main>
        {page === 'leaderboard' && <Leaderboard {...props} />}
        {page === 'explorer' && <QueryExplorer {...props} />}
        {page === 'search' && <LiveSearch dataset={dataset} k={k} retrievers={names} />}
      </main>
    </>
  )
}
