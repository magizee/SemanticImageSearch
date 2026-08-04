import { useState, useCallback, useEffect, useRef } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const RESULT_COUNT = 16
const SHUFFLE_OUT_MS = 420

// COCO's image host is http-only, which browsers can silently fail to load
// as mixed content on our https-served frontend -- route through the
// backend's /image-proxy so images come from our own https origin instead.
const proxiedImageUrl = (url) => url ? `${API_URL}/image-proxy?url=${encodeURIComponent(url)}` : url

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [similarWords, setSimilarWords] = useState([])
  const [phase, setPhase] = useState('initial') // initial | shuffling | results | error
  const [errorMessage, setErrorMessage] = useState('')
  const [batchId, setBatchId] = useState(0)
  const requestId = useRef(0)

  useEffect(() => {
    let cancelled = false
    fetch(`${API_URL}/random?count=${RESULT_COUNT}`)
      .then((res) => res.json())
      .then((data) => {
        if (cancelled) return
        setResults(data)
        setBatchId((n) => n + 1)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  const runSearch = useCallback(async (searchQuery) => {
    const trimmed = searchQuery.trim()
    if (!trimmed) return

    const thisRequest = ++requestId.current
    setPhase('shuffling')
    setErrorMessage('')

    try {
      const [response] = await Promise.all([
        fetch(`${API_URL}/search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: trimmed, top_k: RESULT_COUNT }),
        }),
        sleep(SHUFFLE_OUT_MS),
      ])

      if (!response.ok) {
        throw new Error(`Search failed (${response.status})`)
      }

      const data = await response.json()
      if (thisRequest !== requestId.current) return // a newer search superseded this one

      setResults(data.results)
      setSimilarWords(data.similar_words)
      setBatchId((n) => n + 1)
      setPhase('results')
    } catch (err) {
      if (thisRequest !== requestId.current) return
      setErrorMessage(err.message || 'Something went wrong')
      setPhase('error')
    }
  }, [])

  const handleSubmit = (e) => {
    e.preventDefault()
    runSearch(query)
  }

  const isShuffling = phase === 'shuffling'
  const leftWords = similarWords.slice(0, Math.ceil(similarWords.length / 2))
  const rightWords = similarWords.slice(Math.ceil(similarWords.length / 2))

  return (
    <div className="page">
      <div className="corner-tag">MS&nbsp;COCO / 82,612</div>

      <header className="hero">
        <p className="ghost-text" aria-hidden="true">SEMANTIC SEARCH</p>
        <h1>Semantic<br />Image Search</h1>
        <p className="subtitle">start by searching for an image</p>
      </header>

      <form className="search-bar" onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="grassy field"
          autoFocus
        />
        <button type="submit" disabled={isShuffling || !query.trim()}>
          {isShuffling ? 'Searching' : 'Search'}
        </button>
      </form>

      <div className="stage">
        <aside className="word-column" aria-hidden="true">
          {leftWords.map((w, i) => (
            <span key={`${batchId}-l-${w}`} style={{ animationDelay: `${i * 35}ms` }}>{w}</span>
          ))}
        </aside>

        <main className="results-area">
          {phase === 'error' && (
            <p className="error">Couldn't complete that search: {errorMessage}</p>
          )}

          {results.length > 0 && (
            <div className="grid">
              {results.map((result, i) => (
                <figure
                  className={`card ${isShuffling ? 'card-out' : 'card-in'}`}
                  key={`${batchId}-${result.image_url ?? i}`}
                  style={{ animationDelay: isShuffling ? `${i * 18}ms` : `${i * 45}ms` }}
                >
                  <img src={proxiedImageUrl(result.image_url)} alt={result.caption ?? 'search result'} loading="lazy" />
                  <figcaption>{result.caption}</figcaption>
                </figure>
              ))}
            </div>
          )}
        </main>

        <aside className="word-column" aria-hidden="true">
          {rightWords.map((w, i) => (
            <span key={`${batchId}-r-${w}`} style={{ animationDelay: `${i * 35}ms` }}>{w}</span>
          ))}
        </aside>
      </div>
    </div>
  )
}

export default App
