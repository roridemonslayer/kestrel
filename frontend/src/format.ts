import type { Evaluation, Named } from './api'

export type PageProps = {
  dataset: string
  k: number
  evaluation: Evaluation | null
  evalError: string
  retrievers: Named[]
}

// Monochrome series colors: the strongest retriever gets white, the rest step down to gray.
export const SHADES: Record<string, string> = {
  'hybrid+rerank': '#f2f2f2',
  hybrid: '#c4c4c4',
  bm25: '#8a8a8a',
  dense: '#555555',
}
export const shade = (name: string) => SHADES[name] ?? '#777777'

export const fmt = (n: number) => n.toFixed(3)
export const ms = (n: number) => (n < 10 ? n.toFixed(1) : Math.round(n).toString())

/** Background for a 0..1 score: near-black for 0, light gray for 1. */
export function cellStyle(score: number) {
  const light = Math.round(18 + score * 200)
  return {
    background: `rgb(${light}, ${light}, ${light})`,
    color: score > 0.55 ? '#000' : '#ddd',
  }
}
