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

// virtual layout canvas (percent-independent units) the packing algorithm
// works in; final positions are converted to percentages of this box
const CLOUD_W = 760
const CLOUD_H = 320

// packs similar_words around the logo, wordle-style: each word placed at a
// random horizontal or vertical orientation, spiraling outward from center
// until it finds a spot that doesn't overlap an already-placed word
const buildWordCloud = (words) => {
  if (words.length === 0) return []

  const n = words.length
  const items = words.map((text, i) => {
    const rank = 1 - i / Math.max(1, n - 1) // 1 = most similar .. 0 = least
    const fontSize = 13 + rank * 27
    const vertical = Math.random() < 0.35
    const charW = fontSize * 0.6
    const textW = text.length * charW + 8
    const textH = fontSize * 1.2
    return {
      text,
      fontSize,
      vertical,
      w: vertical ? textH : textW,
      h: vertical ? textW : textH,
      color: GRAY_SHADES[Math.floor(Math.random() * GRAY_SHADES.length)],
    }
  })

  const placed = []
  const centerX = CLOUD_W / 2
  const centerY = CLOUD_H / 2
  const MARGIN = 6
  const overlaps = (x, y, w, h) => placed.some((p) => (
    x < p.x + p.w + 4 && x + w + 4 > p.x &&
    y < p.y + p.h + 4 && y + h + 4 > p.y
  ))
  // keeps words within the canvas so they can't drift down into the
  // search bar (or sideways/up past the hero area) while spiraling out
  const inBounds = (x, y, w, h) => (
    x >= MARGIN && y >= MARGIN && x + w <= CLOUD_W - MARGIN && y + h <= CLOUD_H - MARGIN
  )

  items.forEach((item) => {
    let x = centerX - item.w / 2
    let y = centerY - item.h / 2
    let angle = Math.random() * Math.PI * 2
    let radius = 0
    let attempts = 0
    while ((!inBounds(x, y, item.w, item.h) || overlaps(x, y, item.w, item.h)) && attempts < 500) {
      angle += 0.45
      radius += 1.6
      x = centerX + radius * Math.cos(angle) - item.w / 2
      y = centerY + radius * Math.sin(angle) * 0.6 - item.h / 2
      attempts++
    }
    // fallback: clamp inside the canvas even if a perfectly free spot
    // was never found within the attempt budget
    x = Math.min(Math.max(x, MARGIN), CLOUD_W - MARGIN - item.w)
    y = Math.min(Math.max(y, MARGIN), CLOUD_H - MARGIN - item.h)
    placed.push({ ...item, x, y })
  })

  return placed.map((p, i) => ({
    text: p.text,
    style: {
      left: `${((p.x + p.w / 2) / CLOUD_W) * 100}%`,
      top: `${((p.y + p.h / 2) / CLOUD_H) * 100}%`,
      fontSize: `${p.fontSize}px`,
      color: p.color,
      transform: `translate(-50%, -50%) rotate(${p.vertical ? -90 : 0}deg)`,
      animationDelay: `${(i / placed.length) * 350}ms`,
    },
  }))
}

function App() {
  const [query, setQuery] = useState('')
  const [slots, setSlots] = useState([]) // [{ key, faceA, faceB, flipped }]
  const [wordCloud, setWordCloud] = useState([])
  const [isSearching, setIsSearching] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
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
    setHasSearched(true)
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
          {!hasSearched && <p className="subtitle">start by searching for an image</p>}
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
            {slots.map((slot) => (
              <div className="card" key={slot.key}>
                <div className="flip-card">
                  <div className={`flip-inner ${slot.flipped ? 'is-flipped' : ''}`}>
                    <div className="flip-face flip-front">
                      {slot.faceA && (
                        <img src={proxiedImageUrl(slot.faceA.image_url)} alt={slot.faceA.caption ?? ''} loading="lazy" />
                      )}
                    </div>
                    <div className="flip-face flip-back">
                      {slot.faceB && (
                        <img src={proxiedImageUrl(slot.faceB.image_url)} alt={slot.faceB.caption ?? ''} loading="lazy" />
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
