import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import './HistoryPanel.css'

const POLL_INTERVAL = 3000 // 3 seconds

function HistoryPanel() {
  const [sessions, setSessions] = useState([])
  const [selectedSession, setSelectedSession] = useState(null)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const listRef = useRef(null)

  // Parse "User: ...\nAssistant: ..." pairs from the raw history string
  function parseHistory(raw) {
    return raw.split('\n\n').filter(Boolean).map(block => {
      const lines = block.split('\n')
      const userLine = lines.find(l => l.startsWith('User: '))
      const asstIndex = lines.findIndex(l => l.startsWith('Assistant: '))
      return {
        query: userLine ? userLine.slice(6) : '',
        response: asstIndex !== -1
          ? lines.slice(asstIndex).map(l => l.startsWith('Assistant: ') ? l.slice(11) : l).join('\n').trim()
          : '',
      }
    })
  }

  // Poll session list every POLL_INTERVAL ms
  useEffect(() => {
    let active = true

    function fetchSessions() {
      fetch('/history')
        .then(res => res.json())
        .then(data => {
          if (!active) return
          const ids = data.session_ids || []
          setSessions(prev => {
            // Only update if the list actually changed
            if (prev.length === ids.length && prev.every((id, i) => id === ids[i])) {
              return prev
            }
            return ids
          })
          // Auto-select the first session if none selected or current is gone
          setSelectedSession(prev => {
            if (ids.length === 0) return null
            if (prev && ids.includes(prev)) return prev
            return ids[0]
          })
          setLoading(false)
        })
        .catch(() => { /* backend not ready yet */ })
    }

    fetchSessions()
    const interval = setInterval(fetchSessions, POLL_INTERVAL)
    return () => { active = false; clearInterval(interval) }
  }, [])

  // Poll selected session history every POLL_INTERVAL ms
  useEffect(() => {
    if (!selectedSession) {
      setHistory([])
      return
    }

    let active = true

    function fetchHistory() {
      fetch(`/history?session_id=${encodeURIComponent(selectedSession)}`)
        .then(res => res.json())
        .then(data => {
          if (!active) return
          const parsed = parseHistory(data.history || '')
          setHistory(prev => {
            // Only update if content actually changed
            if (prev.length === parsed.length &&
                prev.every((item, i) => item.query === parsed[i].query && item.response === parsed[i].response)) {
              return prev
            }
            return parsed
          })
        })
        .catch(() => { /* backend not ready yet */ })
    }

    fetchHistory()
    const interval = setInterval(fetchHistory, POLL_INTERVAL)
    return () => { active = false; clearInterval(interval) }
  }, [selectedSession])

  // Auto-scroll to bottom when new history arrives
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight
    }
  }, [history.length])

  if (loading) {
    return (
      <div className="history-panel">
        <h2>History</h2>
        <p className="subtitle">Loading sessions...</p>
      </div>
    )
  }

  return (
    <div className="history-panel">
      {/* Sidebar overlay */}
      {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />}

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
        <div className="sidebar-header">
          <h3>Sessions</h3>
          <button className="sidebar-close" onClick={() => setSidebarOpen(false)}>✕</button>
        </div>
        <nav className="sidebar-list">
          {sessions.length === 0 ? (
            <p className="sidebar-empty">No sessions yet</p>
          ) : (
            sessions.map(id => (
              <button
                key={id}
                className={`sidebar-item ${id === selectedSession ? 'sidebar-item--active' : ''}`}
                onClick={() => { setSelectedSession(id); setSidebarOpen(false) }}
              >
                {id}
              </button>
            ))
          )}
        </nav>
      </aside>

      {/* Hamburger button */}
      <button
        className="hamburger"
        onClick={() => setSidebarOpen(true)}
        aria-label="Open sessions"
      >≡</button>

      <h2>History</h2>
      <p className="subtitle">Current session queries and responses</p>

      {/* Session selector — kept for non-sidebar use but hidden when sidebar shows */}
      {sessions.length > 1 && (
        <div className="session-selector">
          <label htmlFor="session-select">Session: </label>
          <select
            id="session-select"
            value={selectedSession || ''}
            onChange={e => setSelectedSession(e.target.value)}
          >
            {sessions.map(id => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
        </div>
      )}

      {history.length === 0 ? (
        <div className="history-empty">
          <div className="empty-icon">📋</div>
          <h3>No records yet</h3>
          <p>Conversations will appear here once you start chatting</p>
        </div>
      ) : (
        <div className="history-list" ref={listRef}>
          {history.map((item, index) => (
            <div key={index} className="history-item">
              <div className="query">
                <span className="query-label">You</span>
                <p className="query-text">{item.query}</p>
              </div>
              <div className="response">
                <span className="response-label">DevWhisper</span>
                <p className="response-text">{item.response}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      <Link to="/" className="back-link">← Back to Home</Link>
    </div>
  )
}

export default HistoryPanel
