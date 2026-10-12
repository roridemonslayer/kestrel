export function Loading({ evalError, dataset }: { evalError: string; dataset: string }) {
  if (evalError) return <div className="status error">{evalError}</div>
  return (
    <div className="status">
      Running every {dataset} query through every retriever…
      <div className="mono">the first run embeds the corpus; later runs read the cache</div>
    </div>
  )
}
