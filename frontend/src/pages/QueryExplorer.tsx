import { Fragment, useEffect, useMemo, useState } from 'react'
import { api, label, type Doc, type SearchResponse } from '../api'
import { Loading } from '../Loading'
import { cellStyle, fmt, type PageProps } from '../format'

const METRICS = [
  { id: 'ndcg', label: 'nDCG' },
  { id: 'recall', label: 'Recall' },
  { id: 'rr', label: 'RR' },
] as const
type Metric = (typeof METRICS)[number]['id']

const FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'a', label: 'A wins' },
  { id: 'b', label: 'B wins' },
  { id: 'miss', label: 'All miss' },
  { id: 'tie', label: 'Tie' },
] as const
type Filter = (typeof FILTERS)[number]['id']

type Row = { id: string; text: string; scores: Record<string, number>; delta: number }

/** Quote a CSV field when it contains a comma, quote or newline (query texts often do). */
const csvField = (value: string | number) => {
  const text = String(value)
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

function downloadCsv(filename: string, header: string[], rows: (string | number)[][]) {
  const csv = [header, ...rows].map((r) => r.map(csvField).join(',')).join('\n')
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
  const link = Object.assign(document.createElement('a'), { href: url, download: filename })
  link.click()
  URL.revokeObjectURL(url)
}

export default function QueryExplorer({ dataset, k, evaluation, evalError }: PageProps) {
  const names = evaluation?.retrievers.map((r) => r.name) ?? []
  const [metric, setMetric] = useState<Metric>('ndcg')
  const [a, setA] = useState('dense')
  const [b, setB] = useState('bm25')
  const [filter, setFilter] = useState<Filter>('all')
  const [sortBy, setSortBy] = useState<string>('delta')
  const [desc, setDesc] = useState(true)
  const [open, setOpen] = useState<string | null>(null)

  const rows = useMemo<Row[]>(() => {
    if (!evaluation) return []
    const byRetriever = Object.fromEntries(
      evaluation.retrievers.map((r) => [r.name, Object.fromEntries(r.per_query.map((q) => [q.query_id, q[metric]]))]),
    )
    return Object.entries(evaluation.queries).map(([id, q]) => {
      const scores = Object.fromEntries(evaluation.retrievers.map((r) => [r.name, byRetriever[r.name][id] ?? 0]))
      return { id, text: q.text, scores, delta: (scores[a] ?? 0) - (scores[b] ?? 0) }
    })
  }, [evaluation, metric, a, b])

  const visible = useMemo(() => {
    const kept = rows.filter((r) => {
      const values = Object.values(r.scores)
      if (filter === 'a') return r.delta > 0
      if (filter === 'b') return r.delta < 0
      if (filter === 'miss') return values.every((v) => v === 0)
      if (filter === 'tie') return r.delta === 0
      return true
    })
    const key = (r: Row) => (sortBy === 'delta' ? r.delta : sortBy === 'query' ? r.text : r.scores[sortBy])
    return kept.sort((x, y) => {
      const [p, q] = [key(x), key(y)]
      const cmp = typeof p === 'string' ? p.localeCompare(q as string) : (p as number) - (q as number)
      return desc ? -cmp : cmp
    })
  }, [rows, filter, sortBy, desc])

  const counts = useMemo(
    () => ({
      a: rows.filter((r) => r.delta > 0).length,
      b: rows.filter((r) => r.delta < 0).length,
      miss: rows.filter((r) => Object.values(r.scores).every((v) => v === 0)).length,
    }),
    [rows],
  )

  const sort = (key: string) => {
    if (sortBy === key) setDesc(!desc)
    else {
      setSortBy(key)
      setDesc(key !== 'query')
    }
  }
  const arrow = (key: string) => (sortBy === key ? (desc ? ' ↓' : ' ↑') : '')
  const metricLabel = METRICS.find((m) => m.id === metric)!.label

  return (
    <>
      <h1>Query explorer</h1>
      <p className="lede">
        Every query with each retriever's score. Pick two retrievers as A and B to find the queries where one wins and
        the other fails, then click a row to see what each one actually returned.
      </p>
      {!evaluation ? (
        <div className="panel">
          <Loading evalError={evalError} dataset={dataset} />
        </div>
      ) : (
        <div className="panel">
          <div className="filters">
            <div className="pills">
              {METRICS.map((m) => (
                <button key={m.id} className={`pill ${metric === m.id ? 'on' : ''}`} onClick={() => setMetric(m.id)}>
                  {m.label}@{k}
                </button>
              ))}
            </div>
            <label className="control">
              A
              <select value={a} onChange={(e) => setA(e.target.value)}>
                {names.map((n) => (
                  <option key={n} value={n}>
                    {label(n)}
                  </option>
                ))}
              </select>
            </label>
            <label className="control">
              B
              <select value={b} onChange={(e) => setB(e.target.value)}>
                {names.map((n) => (
                  <option key={n} value={n}>
                    {label(n)}
                  </option>
                ))}
              </select>
            </label>
            <div className="pills">
              {FILTERS.map((f) => (
                <button key={f.id} className={`pill ${filter === f.id ? 'on' : ''}`} onClick={() => setFilter(f.id)}>
                  {f.label}
                  {f.id === 'a' || f.id === 'b' || f.id === 'miss' ? ` ${counts[f.id]}` : ''}
                </button>
              ))}
            </div>
            <span className="muted" style={{ marginLeft: 'auto', fontSize: 12 }}>
              {visible.length} of {rows.length} queries
            </span>
            <button
              className="btn"
              onClick={() =>
                downloadCsv(
                  `${dataset}-${metric}@${k}-${filter}.csv`,
                  ['query_id', 'query', ...names.map((n) => `${n}_${metric}@${k}`), `${a}_minus_${b}`],
                  visible.map((r) => [r.id, r.text, ...names.map((n) => r.scores[n].toFixed(4)), r.delta.toFixed(4)]),
                )
              }
            >
              Download CSV
            </button>
          </div>
          <table>
            <thead>
              <tr>
                <th className={`sortable ${sortBy === 'query' ? 'sorted' : ''}`} onClick={() => sort('query')}>
                  Query{arrow('query')}
                </th>
                {names.map((n) => (
                  <th key={n} className={`num sortable ${sortBy === n ? 'sorted' : ''}`} onClick={() => sort(n)}>
                    {label(n)}
                    {arrow(n)}
                  </th>
                ))}
                <th className={`num sortable ${sortBy === 'delta' ? 'sorted' : ''}`} onClick={() => sort('delta')}>
                  {label(a)} − {label(b)}
                  {arrow('delta')}
                </th>
              </tr>
            </thead>
            <tbody>
              {visible.map((r) => (
                <Fragment key={r.id}>
                  <tr className={`qrow ${open === r.id ? 'open' : ''}`} onClick={() => setOpen(open === r.id ? null : r.id)}>
                    <td className="qtext">
                      <span className="dim mono">{r.id.padStart(4, ' ')} </span>
                      {r.text}
                    </td>
                    {names.map((n) => (
                      <td key={n} className="num">
                        <span className="cell mono" style={cellStyle(r.scores[n])}>
                          {fmt(r.scores[n])}
                        </span>
                      </td>
                    ))}
                    <td className={`num mono ${r.delta === 0 ? 'dim' : r.delta > 0 ? 'best' : 'muted'}`}>
                      {r.delta > 0 ? '+' : ''}
                      {fmt(r.delta)}
                    </td>
                  </tr>
                  {open === r.id && (
                    <tr className="detail">
                      <td colSpan={names.length + 2}>
                        <Detail dataset={dataset} k={k} queryId={r.id} text={r.text} names={names} relevant={evaluation.queries[r.id].relevant} metricLabel={metricLabel} scores={r.scores} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
          {!visible.length && <div className="status">No queries match this filter.</div>}
        </div>
      )}
    </>
  )
}

function Detail(props: {
  dataset: string
  k: number
  queryId: string
  text: string
  names: string[]
  relevant: string[]
  metricLabel: string
  scores: Record<string, number>
}) {
  const { dataset, k, queryId, text, names, relevant } = props
  const [data, setData] = useState<SearchResponse | null>(null)
  const [missed, setMissed] = useState<Record<string, Doc>>({})
  const [error, setError] = useState('')

  useEffect(() => {
    api
      .search(text, dataset, names, k, queryId)
      .then((res) => {
        setData(res)
        // Relevant documents that no retriever found: fetch their titles so you can see what was missed.
        const found = new Set(res.columns.flatMap((c) => c.results.map((h) => h.doc_id)))
        const notFound = relevant.filter((id) => !found.has(id))
        if (notFound.length) api.docs(dataset, notFound).then(setMissed).catch(() => {})
      })
      .catch((e: Error) => setError(e.message))
  }, [dataset, k, queryId, text, names, relevant])

  if (error) return <div className="status error">{error}</div>
  if (!data) return <div className="status">Loading results…</div>
  return (
    <div>
      <div className="columns" style={{ ['--n' as string]: data.columns.length }}>
        {data.columns.map((col) => (
          <div key={col.retriever}>
            <div className="col-head">
              {label(col.retriever)}
              <span className="mono">
                {props.metricLabel} {fmt(props.scores[col.retriever])}
              </span>
            </div>
            {col.results.map((h, i) => (
              <div key={h.doc_id} className={`hit ${h.relevant ? 'rel' : ''}`}>
                <span className="rank mono">{i + 1}</span>
                <span className="title">{h.title || h.snippet || h.doc_id}</span>
                <span className="meta mono">
                  {h.relevant && <span className="tag">RELEVANT</span>}
                  <span>score {h.score.toFixed(3)}</span>
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
      {Object.keys(missed).length > 0 && (
        <div style={{ padding: '0 18px 16px' }}>
          <div className="missing">
            <strong>Relevant but missed by every retriever:</strong>
            {Object.entries(missed).map(([id, d]) => (
              <div key={id} style={{ marginTop: 6 }}>
                <span className="mono dim">{id}</span> {d.title || d.snippet}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
