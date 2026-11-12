
import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import MicButton from './MicButton.jsx'
import Markdown from './Markdown.jsx'
import './HistoryPanel.css'

const POLL_INTERVAL = 3000

function highlightText(text = '', highlight = '') {
  if (!highlight.trim()) return text

  const escaped = highlight.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = text.split(new RegExp(`(${escaped})`, 'gi'))

  return (
    <>
      {parts.map((part, index) =>
        part.toLowerCase() === highlight.toLowerCase()
          ? <mark key={index} className="search-highlight">{part}</mark>
          : part
      )}
    </>
  )
}

function truncate(text, max = 48) {
  if (!text) return ''

  const value = String(text).replace(/\s+/g, ' ').trim()
  return value.length > max
    ? `${value.slice(0, max).trimEnd()}…`
    : value
}

function normalizeSessions(list = []) {
  if (!Array.isArray(list)) return []

  return list
    .filter(item => typeof item === 'string' || item?.session_id)
    .map(item => {
      if (typeof item === 'string') {
        return {
          session_id: item,
          last_used: 0,
          message_count: 0,
          preview: ''
        }
      }

      return {
        session_id: String(item.session_id),
        last_used: Number(item.last_used) || 0,
        message_count: Number(item.message_count) || 0,
        preview: String(item.preview || '')
      }
    })
}

function sessionTitle(session) {
  return (
    truncate(session?.preview) ||
    truncate(session?.session_id, 36) ||
    'Untitled session'
  )
}

function formatRelativeTime(timestamp) {
  if (!timestamp || !Number.isFinite(Number(timestamp))) return ''

  const diff = Math.max(0, Date.now() - Number(timestamp) * 1000)
  const minutes = Math.floor(diff / 60000)

  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`

  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`

  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}d ago`

  return new Date(Number(timestamp) * 1000).toLocaleDateString(
    undefined,
    { month: 'short', day: 'numeric' }
  )
}

function parseHistory(historyList) {
  if (!Array.isArray(historyList)) return []

  return historyList.map(entry => {
    if (typeof entry === 'object' && entry !== null) {
      return {
        query: String(entry.query || entry.user || ''),
        response: String(entry.response || entry.assistant || '')
      }
    }

    if (typeof entry !== 'string') {
      return { query: '', response: '' }
    }

    const lines = entry.split('\n')
    const userIndex = lines.findIndex(line => line.startsWith('User: '))
    const assistantIndex = lines.findIndex(
      line => line.startsWith('Assistant: ')
    )

    return {
      query: userIndex >= 0 ? lines[userIndex].slice(6) : '',
      response: assistantIndex >= 0
        ? lines.slice(assistantIndex)
            .map(line =>
              line.startsWith('Assistant: ')
                ? line.slice(11)
                : line
            )
            .join('\n')
            .trim()
        : ''
    }
  })
}

async function readJsonResponse(response) {
  if (!response.ok) {
    let message = `Request failed: HTTP ${response.status}`

    try {
      const data = await response.json()
      message = data.detail || data.message || message
    } catch {
      // Keep the HTTP error message.
    }

    throw new Error(message)
  }

  return response.json()
}

function HistoryPanel() {
  const [searchParams, setSearchParams] = useSearchParams()
  const urlSessionId = searchParams.get('session_id')

  const listRef = useRef(null)
  const stickToBottomRef = useRef(true)
  const streamingRef = useRef(false)
  const streamControllerRef = useRef(null)
  const recognitionRef = useRef(null)
  const feedbackTimerRef = useRef(null)
  const toastTimerRef = useRef(null)
  const copiedTimerRef = useRef(null)
  const copyTimerRef = useRef(null)
  const shareTimerRef = useRef(null)
  const activeSessionRef = useRef(null)

  const [sessions, setSessions] = useState([])
  const [selectedSession, setSelectedSession] = useState(urlSessionId)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [shareFeedback, setShareFeedback] = useState('')
  const [copiedIndex, setCopiedIndex] = useState(null)
  const [copyFeedback, setCopyFeedback] = useState(null)
  const [feedbackMap, setFeedbackMap] = useState({})
  const [showThankYou, setShowThankYou] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [queryText, setQueryText] = useState('')
  const [sending, setSending] = useState(false)
  const [streamError, setStreamError] = useState('')
  const [isListening, setIsListening] = useState(false)
  const [speechSupported, setSpeechSupported] = useState(false)

  const currentSession = sessions.find(
    session => session.session_id === selectedSession
  )

  const filteredSessions = sessions
    .filter(session =>
      `${session.session_id} ${session.preview}`
        .toLowerCase()
        .includes(searchQuery.toLowerCase())
    )
    .sort((a, b) => b.last_used - a.last_used)

  useEffect(() => {
    activeSessionRef.current = selectedSession
  }, [selectedSession])

  // Fetch sessions. This function is reusable by polling and import.
  const fetchSessions = useCallback(async () => {
    const response = await fetch('/history')
    const data = await readJsonResponse(response)

    const ids = Array.isArray(data.session_ids)
      ? data.session_ids.map(String)
      : []

    const normalized = normalizeSessions(
      Array.isArray(data.sessions) && data.sessions.length
        ? data.sessions
        : ids
    )

    setSessions(normalized)

    const availableIds = normalized.map(session => session.session_id)

    setSelectedSession(previous => {
      if (urlSessionId && availableIds.includes(urlSessionId)) {
        return urlSessionId
      }

      if (previous && availableIds.includes(previous)) {
        return previous
      }

      return availableIds[0] || null
    })

    return normalized
  }, [urlSessionId])

  // Keep URL synchronized with selected session.
  const selectSession = useCallback((id) => {
    if (!id) return

    setSelectedSession(id)
    setSearchParams({ session_id: id })
    setSidebarOpen(false)
    setStreamError('')
  }, [setSearchParams])

  // Initial session loading and polling.
  useEffect(() => {
    let active = true

    const load = async () => {
      try {
        await fetchSessions()
      } catch (error) {
        if (active) {
          console.error('Failed to load sessions:', error)
        }
      } finally {
        if (active) setLoading(false)
      }
    }

    load()

    const interval = setInterval(() => {
      if (active) {
        fetchSessions().catch(error =>
          console.error('Session polling failed:', error)
        )
      }
    }, POLL_INTERVAL)

    return () => {
      active = false
      clearInterval(interval)
    }
  }, [fetchSessions])

  // Load history for the selected session.
  useEffect(() => {
    if (!selectedSession) {
      setHistory([])
      return
    }

    let active = true
    const sessionId = selectedSession

    const fetchHistory = async () => {
      try {
        const response = await fetch(
          `/history?session_id=${encodeURIComponent(sessionId)}`
        )

        const data = await readJsonResponse(response)

        if (!active || streamingRef.current) return
        if (activeSessionRef.current !== sessionId) return

        const parsed = parseHistory(data.history || [])
        setHistory(parsed)
      } catch (error) {
        if (active) {
          console.error('History polling failed:', error)
        }
      }
    }

    fetchHistory()

    const interval = setInterval(fetchHistory, POLL_INTERVAL)

    return () => {
      active = false
      clearInterval(interval)
    }
  }, [selectedSession])

  // Auto-scroll when new messages arrive.
  useEffect(() => {
    if (!stickToBottomRef.current) return

    const element = listRef.current

    if (element) {
      element.scrollTop = element.scrollHeight
    }
  }, [history])

  const handleScroll = () => {
    const element = listRef.current
    if (!element) return

    stickToBottomRef.current =
      element.scrollHeight -
      element.scrollTop -
      element.clientHeight < 120
  }

  // Copy helper with clipboard fallback.
  const copyText = async text => {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return
    }

    const textarea = document.createElement('textarea')
    textarea.value = text
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'

    document.body.appendChild(textarea)
    textarea.select()

    const copied = document.execCommand('copy')
    document.body.removeChild(textarea)

    if (!copied) {
      throw new Error('Clipboard copy failed')
    }
  }

  const showShareMessage = message => {
    setShareFeedback(message)

    if (shareTimerRef.current) {
      clearTimeout(shareTimerRef.current)
    }

    shareTimerRef.current = setTimeout(() => {
      setShareFeedback('')
      shareTimerRef.current = null
    }, 2500)
  }

  const handleShareExchange = async (query, response, index) => {
    const text = `You: ${query}\nAssistant: ${response}`

    try {
      if (navigator.share) {
        await navigator.share({
          title: 'Share DevWhisper Conversation',
          text
        })
      } else {
        await copyText(text)
      }

      setCopiedIndex(index)

      if (copiedTimerRef.current) {
        clearTimeout(copiedTimerRef.current)
      }

      copiedTimerRef.current = setTimeout(() => {
        setCopiedIndex(null)
      }, 2000)
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.error('Share failed:', error)
        showShareMessage('Unable to share conversation')
      }
    }
  }

  const handleShareSessionLink = async () => {
    if (!selectedSession) {
      showShareMessage('Please select a session to share')
      return
    }

    const url =
      `${window.location.origin}/history?session_id=${encodeURIComponent(selectedSession)}`

    try {
      await copyText(url)
      showShareMessage('Share link copied to clipboard')
    } catch (error) {
      console.error('Share link copy failed:', error)
      showShareMessage('Failed to copy share link')
    }
  }

  const handleCopy = async (response, index) => {
    try {
      await copyText(response)
      setCopyFeedback(index)

      if (copyTimerRef.current) {
        clearTimeout(copyTimerRef.current)
      }

      copyTimerRef.current = setTimeout(() => {
        setCopyFeedback(null)
      }, 2000)
    } catch (error) {
      console.error('Copy failed:', error)
      showShareMessage('Failed to copy response')
    }
  }

  const handleFeedback = type => index => {
    setFeedbackMap(previous => ({
      ...previous,
      [index]: previous[index] === type ? null : type
    }))

    setShowThankYou(true)

    if (feedbackTimerRef.current) {
      clearTimeout(feedbackTimerRef.current)
    }

    feedbackTimerRef.current = setTimeout(() => {
      setShowThankYou(false)
      feedbackTimerRef.current = null
    }, 2500)
  }

  const handleLike = handleFeedback('like')
  const handleDislike = handleFeedback('dislike')

  // Fallback endpoint.
  const webhookFallback = async (query, sessionId) => {
    const toolId = window.crypto?.randomUUID?.() ||
      `${Date.now()}-${Math.random().toString(36).slice(2)}`

    const payload = {
      message: {
        type: 'tool-calls',
        sessionId,
        toolCalls: [{
          id: toolId,
          type: 'function',
          function: {
            name: 'query_codebase',
            arguments: JSON.stringify({ query })
          }
        }]
      }
    }

    const response = await fetch('/webhook', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })

    const data = await readJsonResponse(response)

    if (data.status === 'error') {
      throw new Error(data.message || 'Codebase query failed')
    }

    return data.results?.[0]?.result ||
      'No response was generated by the codebase assistant.'
  }

  // Send a follow-up and stream its response.
  const submitQuery = async (query, sessionId) => {
    if (streamingRef.current || !query.trim() || !sessionId) return

    streamingRef.current = true
    stickToBottomRef.current = true
    setSending(true)
    setStreamError('')

    const controller = new AbortController()
    streamControllerRef.current = controller

    try {
      const response = await fetch('/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          sessionId
        }),
        signal: controller.signal
      })

      // Use fallback only when the streaming endpoint is unavailable.
      if (response.status === 404) {
        const fallbackResult = await webhookFallback(query, sessionId)

        setHistory(previous => {
          const copy = [...previous]
          const last = copy[copy.length - 1]

          if (last) {
            copy[copy.length - 1] = {
              ...last,
              response: fallbackResult
            }
          }

          return copy
        })

        return
      }

      if (!response.ok) {
        throw new Error(`Streaming failed: HTTP ${response.status}`)
      }

      if (!response.body) {
        throw new Error('Streaming response body is unavailable')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let accumulated = ''

      while (true) {
        const { value, done } = await reader.read()

        if (done) break

        accumulated += decoder.decode(value, { stream: true })

        setHistory(previous => {
          const copy = [...previous]
          const last = copy[copy.length - 1]

          if (last) {
            copy[copy.length - 1] = {
              ...last,
              response: accumulated
            }
          }

          return copy
        })

        if (stickToBottomRef.current && listRef.current) {
          listRef.current.scrollTop = listRef.current.scrollHeight
        }
      }

      accumulated += decoder.decode()

      if (accumulated) {
        setHistory(previous => {
          const copy = [...previous]
          const last = copy[copy.length - 1]

          if (last) {
            copy[copy.length - 1] = {
              ...last,
              response: accumulated
            }
          }

          return copy
        })
      }
    } catch (error) {
      if (error.name === 'AbortError') return

      console.error('Streaming failed:', error)

      setStreamError(
        error.message || 'Failed to reach the DevWhisper backend.'
      )

      setHistory(previous => {
        const copy = [...previous]
        const last = copy[copy.length - 1]

        if (last && !last.response) {
          copy[copy.length - 1] = {
            ...last,
            response: 'Unable to get a response. Please try again.'
          }
        }

        return copy
      })
    } finally {
      streamingRef.current = false
      streamControllerRef.current = null
      setSending(false)
    }
  }

  const handleSubmit = event => {
    event?.preventDefault()

    const query = queryText.trim()

    if (!selectedSession || sending || !query) return

    setQueryText('')
    setStreamError('')

    setHistory(previous => [
      ...previous,
      { query, response: '' }
    ])

    submitQuery(query, selectedSession)
  }

  // Speech recognition setup.
  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition

    if (!SpeechRecognition) {
      setSpeechSupported(false)
      return
    }

    setSpeechSupported(true)

    const recognition = new SpeechRecognition()
    recognition.continuous = false
    recognition.interimResults = true
    recognition.lang = 'en-US'

    recognition.onstart = () => {
      setIsListening(true)
      setStreamError('')
    }

    recognition.onresult = event => {
      let transcript = ''

      for (let index = 0; index < event.results.length; index++) {
        transcript += event.results[index][0].transcript
      }

      setQueryText(transcript)
    }

    recognition.onerror = event => {
      console.error('Speech recognition error:', event.error)
      setIsListening(false)

      if (event.error !== 'aborted' && event.error !== 'no-speech') {
        setStreamError('Microphone recognition failed. Please try again.')
      }
    }

    recognition.onend = () => {
      setIsListening(false)
    }

    recognitionRef.current = recognition

    return () => {
      recognition.abort()
      recognitionRef.current = null
    }
  }, [])

  const handleMicClick = () => {
    if (!speechSupported || !recognitionRef.current) {
      setStreamError('Speech recognition is not supported in this browser.')
      return
    }

    if (isListening) {
      recognitionRef.current.stop()
      return
    }

    setQueryText('')
    setStreamError('')

    try {
      recognitionRef.current.start()
    } catch (error) {
      console.error('Could not start speech recognition:', error)
      setStreamError('Could not activate microphone. Please try again.')
    }
  }

  // Export session.
  const handleExportSession = async () => {
    if (!selectedSession) return

    try {
      const response = await fetch(
        `/history/export/${encodeURIComponent(selectedSession)}`
      )

      const data = await readJsonResponse(response)

      const blob = new Blob(
        [JSON.stringify(data.session, null, 2)],
        { type: 'application/json' }
      )

      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')

      anchor.href = url
      anchor.download = `devwhisper-session-${selectedSession}.json`

      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()

      URL.revokeObjectURL(url)
      showShareMessage('Session exported successfully')
    } catch (error) {
      console.error('Export failed:', error)
      showShareMessage('Failed to export session')
    }
  }

  // Delete session.
  const handleDeleteSession = async (sessionId, event) => {
    event?.stopPropagation()

    if (!window.confirm(`Delete conversation session '${sessionId}'?`)) {
      return
    }

    try {
      const response = await fetch(
        `/history/${encodeURIComponent(sessionId)}`,
        { method: 'DELETE' }
      )

      await readJsonResponse(response)

      const remaining = sessions.filter(
        session => session.session_id !== sessionId
      )

      setSessions(remaining)

      if (selectedSession === sessionId) {
        const nextSession = remaining[0]?.session_id || null

        setSelectedSession(nextSession)

        if (nextSession) {
          setSearchParams({ session_id: nextSession })
        } else {
          setSearchParams({})
          setHistory([])
        }
      }

      showShareMessage('Session deleted')
    } catch (error) {
      console.error('Delete failed:', error)
      showShareMessage(error.message || 'Failed to delete session')
    }
  }

  // Import session JSON.
  const handleImportFile = async event => {
    const file = event.target.files?.[0]
    if (!file) return

    try {
      if (!file.name.toLowerCase().endsWith('.json')) {
        throw new Error('Please select a JSON file')
      }

      const sessionData = JSON.parse(await file.text())

      if (!sessionData || typeof sessionData !== 'object') {
        throw new Error('Invalid session JSON')
      }

      const response = await fetch('/history/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session: sessionData })
      })

      const data = await readJsonResponse(response)

      if (!data.session_id) {
        throw new Error('Import response did not contain a session ID')
      }

      await fetchSessions()
      selectSession(data.session_id)

      showShareMessage('Session imported successfully')
    } catch (error) {
      console.error('Import failed:', error)
      window.alert(`Session import error: ${error.message}`)
    } finally {
      event.target.value = ''
    }
  }

  // Clean up timers and active streaming request.
  useEffect(() => {
    return () => {
      streamControllerRef.current?.abort()

      if (feedbackTimerRef.current) clearTimeout(feedbackTimerRef.current)
      if (toastTimerRef.current) clearTimeout(toastTimerRef.current)
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current)
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current)
      if (shareTimerRef.current) clearTimeout(shareTimerRef.current)
    }
  }, [])

  if (loading) {
    return (
      <div className="history-panel">
        <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
          <div className="sidebar-header">
            <h3>Sessions</h3>
            <button
              className="sidebar-close"
              onClick={() => setSidebarOpen(false)}
            >
              ✕
            </button>
          </div>
          <nav className="sidebar-list">
            <p className="sidebar-empty">Loading sessions...</p>
          </nav>
        </aside>

        <header className="chat-header">
          <button
            className="hamburger"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open sessions"
          >
            ≡
          </button>
          <div className="chat-header-info">
            <h2>History</h2>
            <p className="subtitle">Loading conversations...</p>
          </div>
          <Link to="/" className="back-link">← Home</Link>
        </header>

        <div className="chat-messages chat-messages--empty">
          <p className="history-loading">Loading sessions...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="history-panel">
      {sidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
        <div className="sidebar-header">
          <h3>Sessions</h3>
          <button
            className="sidebar-close"
            onClick={() => setSidebarOpen(false)}
          >
            ✕
          </button>
        </div>

        <div className="sidebar-search">
          <input
            type="text"
            className="search-input"
            placeholder="Search sessions..."
            value={searchQuery}
            onChange={event => setSearchQuery(event.target.value)}
          />

          {searchQuery && (
            <button
              className="search-clear"
              onClick={() => setSearchQuery('')}
              aria-label="Clear search"
            >
              ✕
            </button>
          )}
        </div>

        <div
          className="sidebar-import-wrapper"
          style={{ padding: '0.5rem 1rem' }}
        >
          <label
            className="import-btn"
            style={{
              display: 'inline-block',
              width: '100%',
              textAlign: 'center',
              cursor: 'pointer',
              padding: '0.4rem',
              border: '1px dashed var(--border-color, #444)',
              borderRadius: '4px',
              fontSize: '0.85rem'
            }}
          >
            📥 Import Session JSON
            <input
              type="file"
              accept=".json,application/json"
              onChange={handleImportFile}
              style={{ display: 'none' }}
            />
          </label>
        </div>

        <nav className="sidebar-list">
          {filteredSessions.length === 0 ? (
            <p className="sidebar-empty">
              {sessions.length === 0
                ? 'No sessions yet'
                : 'No matching sessions'}
            </p>
          ) : (
            filteredSessions.map(session => (
              <div
                key={session.session_id}
                className={`sidebar-item-container ${
                  session.session_id === selectedSession
                    ? 'sidebar-item--active'
                    : ''
                }`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between'
                }}
              >
                <button
                  className="sidebar-item"
                  style={{
                    flex: 1,
                    border: 'none',
                    background: 'transparent',
                    textAlign: 'left'
                  }}
                  onClick={() => selectSession(session.session_id)}
                >
                  <span className="sidebar-item-title">
                    {highlightText(sessionTitle(session), searchQuery)}
                  </span>

                  <span className="sidebar-item-meta">
                    {formatRelativeTime(session.last_used)}
                    {session.message_count > 0 &&
                      ` · ${session.message_count} msg${
                        session.message_count > 1 ? 's' : ''
                      }`}
                  </span>
                </button>

                <button
                  className="session-delete-btn"
                  title="Delete session"
                  aria-label="Delete session"
                  onClick={event =>
                    handleDeleteSession(session.session_id, event)
                  }
                  style={{
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    opacity: 0.6,
                    padding: '0.3rem'
                  }}
                >
                  🗑️
                </button>
              </div>
            ))
          )}
        </nav>
      </aside>

      <header className="chat-header">
        <button
          className="hamburger"
          onClick={() => setSidebarOpen(true)}
          aria-label="Open sessions"
        >
          ≡
        </button>

        <div className="chat-header-info">
          <h2>History</h2>
          <p className="subtitle">
            {currentSession
              ? sessionTitle(currentSession)
              : 'Conversation history'}
          </p>
        </div>

        <div className="header-actions">
          {selectedSession && (
            <button
              className="share-button"
              onClick={handleExportSession}
              title="Export conversation JSON"
            >
              💾 Export JSON
            </button>
          )}

          <button
            className="share-button"
            onClick={handleShareSessionLink}
          >
            🔗 Share link
          </button>

          {shareFeedback && (
            <span className="share-feedback">{shareFeedback}</span>
          )}

          <Link to="/" className="back-link">← Home</Link>
        </div>
      </header>

      {currentSession && (
        <div className="session-detail">
          <span
            className="session-detail-id"
            title={currentSession.session_id}
          >
            {currentSession.session_id}
          </span>

          {currentSession.last_used > 0 && (
            <span className="session-detail-time">
              Last used {formatRelativeTime(currentSession.last_used)}
            </span>
          )}

          {currentSession.message_count > 0 && (
            <span className="session-detail-count">
              {currentSession.message_count} message
              {currentSession.message_count > 1 ? 's' : ''}
            </span>
          )}
        </div>
      )}

      <div
        className="chat-messages"
        ref={listRef}
        onScroll={handleScroll}
      >
        {history.length === 0 ? (
          <div className="history-empty">
            <div className="empty-icon">💬</div>
            <h3>No messages yet</h3>
            <p>Ask something below to start the conversation</p>
          </div>
        ) : (
          history.map((item, index) => (
            <div key={`${selectedSession}-${index}`} className="exchange">
              <div className="chat-message chat-message--user">
                <div className="chat-bubble chat-bubble--user">
                  {item.query}
                </div>
              </div>

              <div className="chat-message chat-message--assistant">
                <div className="chat-bubble chat-bubble--assistant response-body">
                  <Markdown text={item.response} />
                </div>

                <div className="chat-actions">
                  <button
                    className="share-exchange-btn"
                    onClick={() =>
                      handleShareExchange(
                        item.query,
                        item.response,
                        index
                      )
                    }
                    aria-label="Share this exchange"
                  >
                    {copiedIndex === index ? 'Copied!' : 'Share'}
                  </button>

                  <button
                    className={`like-button ${
                      feedbackMap[index] === 'like' ? 'active' : ''
                    }`}
                    onClick={() => handleLike(index)}
                    aria-label="Like response"
                  >
                    👍
                  </button>

                  <button
                    className={`dislike-button ${
                      feedbackMap[index] === 'dislike' ? 'active' : ''
                    }`}
                    onClick={() => handleDislike(index)}
                    aria-label="Dislike response"
                  >
                    👎
                  </button>

                  <button
                    className={`copy-button ${
                      copyFeedback === index ? 'active' : ''
                    }`}
                    onClick={() => handleCopy(item.response, index)}
                    aria-label="Copy response"
                  >
                    {copyFeedback === index ? '✔' : '📋'}
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {streamError && (
        <div className="stream-error">⚠️ {streamError}</div>
      )}

      <form className="chat-input-bar" onSubmit={handleSubmit}>
        <MicButton
          isListening={isListening}
          onClick={handleMicClick}
          disabled={sending || !speechSupported}
        />

        <textarea
          className="chat-input"
          value={queryText}
          placeholder={
            isListening
              ? 'Listening... Speak now.'
              : selectedSession
                ? 'Ask a follow-up...'
                : 'Select a session from the sidebar to start chatting'
          }
          disabled={sending || !selectedSession}
          rows={1}
          onChange={event => {
            setQueryText(event.target.value)
            event.target.style.height = 'auto'
            event.target.style.height =
              `${Math.min(event.target.scrollHeight, 160)}px`
          }}
          onKeyDown={event => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              handleSubmit(event)
            }
          }}
        />

        <button
          type="submit"
          className="send-button"
          disabled={sending || !queryText.trim() || !selectedSession}
        >
          {sending ? 'Sending…' : 'Send'}
        </button>
      </form>

      {showThankYou && (
        <div className="thank-you-toast">
          Thanks for your feedback 🙌
        </div>
      )}
    </div>
  )
}

export default HistoryPanel

