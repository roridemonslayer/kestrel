import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { label } from '../api'
import { Loading } from '../Loading'
import { fmt, ms, shade, type PageProps } from '../format'

const TOOLTIP = {
  contentStyle: { background: '#111', border: '1px solid #2e2e2e', borderRadius: 6, fontSize: 12 },
  labelStyle: { color: '#8a8a8a' },
  cursor: { fill: 'rgba(255,255,255,0.04)' },
}

export default function Leaderboard({ dataset, k, evaluation, evalError, retrievers }: PageProps) {
  const describe = (name: string) => retrievers.find((r) => r.name === name)?.description ?? ''

  return (
    <>
      <h1>Leaderboard</h1>
      <p className="lede">
        Every {dataset} test query, run through each retriever and scored against human relevance judgments.
        Higher is better for the three quality metrics; lower is better for latency.
      </p>
      {!evaluation ? (
        <div className="panel">
          <Loading evalError={evalError} dataset={dataset} />
        </div>
      ) : (
        <Board evaluation={evaluation} k={k} describe={describe} />
      )}
    </>
  )
}

function Board({ evaluation, k, describe }: { evaluation: NonNullable<PageProps['evaluation']>; k: number; describe: (n: string) => string }) {
  const rows = [...evaluation.retrievers].sort((a, b) => b.ndcg - a.ndcg)
  const best = {
    recall: Math.max(...rows.map((r) => r.recall)),
    mrr: Math.max(...rows.map((r) => r.mrr)),
    ndcg: Math.max(...rows.map((r) => r.ndcg)),
    p50: Math.min(...rows.map((r) => r.p50_ms)),
    p95: Math.min(...rows.map((r) => r.p95_ms)),
  }
  const quality = [
    { metric: `recall@${k}`, ...Object.fromEntries(rows.map((r) => [r.name, r.recall])) },
    { metric: `MRR@${k}`, ...Object.fromEntries(rows.map((r) => [r.name, r.mrr])) },
    { metric: `nDCG@${k}`, ...Object.fromEntries(rows.map((r) => [r.name, r.ndcg])) },
  ]
  const latency = [
    { metric: 'p50', ...Object.fromEntries(rows.map((r) => [r.name, r.p50_ms])) },
    { metric: 'p95', ...Object.fromEntries(rows.map((r) => [r.name, r.p95_ms])) },
  ]
  const queryCount = Object.keys(evaluation.queries).length

  return (
    <>
      <div className="panel">
        <div className="panel-head">
          <span>
            {evaluation.dataset} · {queryCount} queries · k = {k}
          </span>
          <span>best in white</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Retriever</th>
              <th className="num">recall@{k}</th>
              <th className="num">MRR@{k}</th>
              <th className="num">nDCG@{k}</th>
              <th className="num">p50 ms</th>
              <th className="num">p95 ms</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name}>
                <td>
                  <span className="swatch" style={{ background: shade(r.name) }} />
                  {label(r.name)}
                  <div className="desc">{describe(r.name)}</div>
                </td>
                <td className={`num mono ${r.recall === best.recall ? 'best' : 'muted'}`}>{fmt(r.recall)}</td>
                <td className={`num mono ${r.mrr === best.mrr ? 'best' : 'muted'}`}>{fmt(r.mrr)}</td>
                <td className={`num mono ${r.ndcg === best.ndcg ? 'best' : 'muted'}`}>{fmt(r.ndcg)}</td>
                <td className={`num mono ${r.p50_ms === best.p50 ? 'best' : 'muted'}`}>{ms(r.p50_ms)}</td>
                <td className={`num mono ${r.p95_ms === best.p95 ? 'best' : 'muted'}`}>{ms(r.p95_ms)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid-2">
        <div className="panel">
          <div className="panel-head">Quality</div>
          <div className="chart">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={quality} barGap={4} barCategoryGap="22%">
                <CartesianGrid stroke="#1f1f1f" vertical={false} />
                <XAxis dataKey="metric" stroke="#555" tickLine={false} axisLine={false} fontSize={12} />
                <YAxis domain={[0, 1]} stroke="#555" tickLine={false} axisLine={false} fontSize={12} width={36} />
                <Tooltip {...TOOLTIP} formatter={(v) => fmt(Number(v))} />
                {rows.map((r) => (
                  <Bar key={r.name} dataKey={r.name} name={label(r.name)} fill={shade(r.name)} radius={[3, 3, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Legend names={rows.map((r) => r.name)} />
        </div>
        <div className="panel">
          <div className="panel-head">Latency per query (ms)</div>
          <div className="chart">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={latency} barGap={4} barCategoryGap="28%">
                <CartesianGrid stroke="#1f1f1f" vertical={false} />
                <XAxis dataKey="metric" stroke="#555" tickLine={false} axisLine={false} fontSize={12} />
                <YAxis stroke="#555" tickLine={false} axisLine={false} fontSize={12} width={36} />
                <Tooltip {...TOOLTIP} formatter={(v) => `${ms(Number(v))} ms`} />
                {rows.map((r) => (
                  <Bar key={r.name} dataKey={r.name} name={label(r.name)} fill={shade(r.name)} radius={[3, 3, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Legend names={rows.map((r) => r.name)} />
        </div>
      </div>

      <div className="panel" style={{ marginTop: 16 }}>
        <div className="panel-head">What the metrics mean</div>
        <table>
          <tbody>
            <tr>
              <td className="mono">recall@{k}</td>
              <td className="muted">Of the documents a human marked relevant, the share that appear anywhere in the top {k}.</td>
            </tr>
            <tr>
              <td className="mono">MRR@{k}</td>
              <td className="muted">1 ÷ the rank of the first relevant result (1.0 if it's first, 0.5 if second…), averaged over queries.</td>
            </tr>
            <tr>
              <td className="mono">nDCG@{k}</td>
              <td className="muted">Rewards relevant results more the higher they rank, scaled so a perfect ordering scores 1.0.</td>
            </tr>
            <tr>
              <td className="mono">p50 / p95</td>
              <td className="muted">Typical and near-worst search time: half of queries finish under p50, 95% under p95.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </>
  )
}

function Legend({ names }: { names: string[] }) {
  return (
    <div className="legend">
      {names.map((n) => (
        <span key={n}>
          <span className="swatch" style={{ background: shade(n) }} />
          {label(n)}
        </span>
      ))}
    </div>
  )
}
