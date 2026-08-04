import { useState, useCallback, useEffect, useRef } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const RESULT_COUNT = 16
const FLIP_WINDOW_MS = 700 // total span over which the 16 cards flip, in random order

// COCO's image host is http-only, which browsers can silently fail to load
// as mixed content on our https-served frontend -- route through the
// backend's /image-proxy so images come from our own https origin instead.
const proxiedImageUrl = (url) => url ? `${API_URL}/image-proxy?url=${encodeURIComponent(url)}` : url

const shuffle = (arr) => {
  const a = [...arr]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

const GRAY_SHADES = ['#0a0a0a', '#2b2b2b', '#4a4a4a', '#6b6b6b', '#8a8a8a']

// scatters similar_words around the logo with randomized size/position/shade
const buildWordCloud = (words) => words.map((text) => ({
  text,
  style: {
    left: `${5 + Math.random() * 90}%`,
    top: `${Math.random() * 100}%`,
    fontSize: `${11 + Math.random() * 24}px`,
    color: GRAY_SHADES[Math.floor(Math.random() * GRAY_SHADES.length)],
    transform: `translate(-50%, -50%) rotate(${(Math.random() - 0.5) * 16}deg)`,
    animationDelay: `${Math.random() * 400}ms`,
  },
}))

function App() {
  const [query, setQuery] = useState('')
  const [slots, setSlots] = useState([]) // [{ key, faceA, faceB, flipped }]
  const [wordCloud, setWordCloud] = useState([])
  const [isSearching, setIsSearching] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
  const requestId = useRef(0)
  const timeouts = useRef([])

  useEffect(() => {
    fetch(`${API_URL}/random?count=${RESULT_COUNT}`)
      .then((res) => res.json())
      .then((data) => {
        setSlots(data.map((result, i) => ({ key: i, faceA: result, faceB: null, flipped: false })))
      })
      .catch(() => {})
  }, [])

  useEffect(() => () => timeouts.current.forEach(clearTimeout), [])

  const runSearch = useCallback(async (searchQuery) => {
    const trimmed = searchQuery.trim()
    if (!trimmed) return

    const thisRequest = ++requestId.current
    timeouts.current.forEach(clearTimeout)
    timeouts.current = []
    setIsSearching(true)
    setErrorMessage('')

    try {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmed, top_k: RESULT_COUNT }),
      })

      if (!response.ok) {
        throw new Error(`Search failed (${response.status})`)
      }

      const data = await response.json()
      if (thisRequest !== requestId.current) return // a newer search superseded this one

      setWordCloud(buildWordCloud(data.similar_words))
      setIsSearching(false)

      // flip each card at its own random moment, in random order, instead
      // of swapping everything at once
      const order = shuffle([...Array(data.results.length).keys()])
      order.forEach((slotIndex, position) => {
        const delay = (position / order.length) * FLIP_WINDOW_MS + Math.random() * 60
        const t = setTimeout(() => {
          setSlots((prev) => prev.map((s, i) => {
            if (i !== slotIndex) return s
            const newResult = data.results[slotIndex]
            return s.flipped
              ? { ...s, faceA: newResult, flipped: false }
              : { ...s, faceB: newResult, flipped: true }
          }))
        }, delay)
        timeouts.current.push(t)
      })
    } catch (err) {
      if (thisRequest !== requestId.current) return
      setErrorMessage(err.message || 'Something went wrong')
      setIsSearching(false)
    }
  }, [])

  const handleSubmit = (e) => {
    e.preventDefault()
    runSearch(query)
  }

  return (
    <div className="page">
      <div className="corner-tag">MS&nbsp;COCO / 82,612</div>

      <div className="hero-wrap">
        <div className="word-cloud" aria-hidden="true">
          {wordCloud.map((w, i) => (
            <span key={`${w.text}-${i}`} style={w.style}>{w.text}</span>
          ))}
        </div>
        <header className="hero">
          <h1>Semantic<br />Image Search</h1>
          <p className="subtitle">start by searching for an image</p>
        </header>
      </div>

      <form className="search-bar" onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="grassy field"
          autoFocus
        />
        <button type="submit" disabled={isSearching || !query.trim()}>
          {isSearching ? 'Searching' : 'Search'}
        </button>
      </form>

      <main className="results-area">
        {errorMessage && (
          <p className="error">Couldn't complete that search: {errorMessage}</p>
        )}

        {slots.length > 0 && (
          <div className="grid">
            {slots.map((slot) => {
              const shownCaption = (slot.flipped ? slot.faceB : slot.faceA)?.caption
              return (
                <figure className="card" key={slot.key}>
                  <div className="flip-card">
                    <div className={`flip-inner ${slot.flipped ? 'is-flipped' : ''}`}>
                      <div className="flip-face flip-front">
                        {slot.faceA && (
                          <img src={proxiedImageUrl(slot.faceA.image_url)} alt="" loading="lazy" />
                        )}
                      </div>
                      <div className="flip-face flip-back">
                        {slot.faceB && (
                          <img src={proxiedImageUrl(slot.faceB.image_url)} alt="" loading="lazy" />
                        )}
                      </div>
                    </div>
                  </div>
                  <figcaption>{shownCaption}</figcaption>
                </figure>
              )
            })}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
