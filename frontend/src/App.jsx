import { useState, useCallback } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [status, setStatus] = useState('idle') // idle | loading | error | done
  const [errorMessage, setErrorMessage] = useState('')

  const runSearch = useCallback(async (searchQuery) => {
    const trimmed = searchQuery.trim()
    if (!trimmed) return

    setStatus('loading')
    setErrorMessage('')

    try {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmed, top_k: 16 }),
      })

      if (!response.ok) {
        throw new Error(`Search failed (${response.status})`)
      }

      const data = await response.json()
      setResults(data)
      setStatus('done')
    } catch (err) {
      setErrorMessage(err.message || 'Something went wrong')
      setStatus('error')
    }
  }, [])

  const handleSubmit = (e) => {
    e.preventDefault()
    runSearch(query)
  }

  return (
    <div className="page">
      <header className="hero">
        <h1>Semantic Image Search</h1>
        <p>Describe an image in words &mdash; search the MS&nbsp;COCO dataset by meaning, not keywords.</p>
      </header>

      <form className="search-bar" onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="a dog running on the beach..."
          autoFocus
        />
        <button type="submit" disabled={status === 'loading' || !query.trim()}>
          {status === 'loading' ? 'Searching…' : 'Search'}
        </button>
      </form>

      <main className="results-area">
        {status === 'idle' && (
          <p className="hint">Try a search above to get started.</p>
        )}

        {status === 'loading' && (
          <div className="grid" aria-hidden="true">
            {Array.from({ length: 8 }).map((_, i) => (
              <div className="card skeleton" key={i} />
            ))}
          </div>
        )}

        {status === 'error' && (
          <p className="error">Couldn't complete that search: {errorMessage}</p>
        )}

        {status === 'done' && results.length === 0 && (
          <p className="hint">No results found.</p>
        )}

        {status === 'done' && results.length > 0 && (
          <div className="grid">
            {results.map((result, i) => (
              <figure className="card" key={result.image_url ?? i}>
                <img src={result.image_url} alt={result.caption ?? 'search result'} loading="lazy" />
                <figcaption>{result.caption}</figcaption>
              </figure>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
