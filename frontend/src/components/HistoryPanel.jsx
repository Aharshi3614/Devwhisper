import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import MicButton from './MicButton.jsx'
import Markdown from './Markdown.jsx'
import './HistoryPanel.css'

const POLL_INTERVAL = 3000 // 3 seconds

function highlightText(text, highlight) {
  if (!highlight.trim()) return text
  const escaped = highlight.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const regex = new RegExp(`(${escaped})`, 'gi')
  const parts = text.split(regex)
  return (
    <>
      {parts.map((part, i) =>
        part.toLowerCase() === highlight.toLowerCase() ? (
          <mark key={i} className="search-highlight">{part}</mark>
        ) : (
          part
        )
      )}
    </>
  )
}

function truncate(text, max = 48) {
  if (!text) return ''
  const singleLine = text.replace(/\s+/g, ' ').trim()
  return singleLine.length > max ? singleLine.slice(0, max).trimEnd() + '…' : singleLine
}

// Normalize session entries into metadata objects. Backward compatible with
// older backends that only return a list of session IDs.
function normalizeSessions(list) {
  return (list || []).map(item =>
    typeof item === 'string'
      ? { session_id: item, last_used: 0, message_count: 0, preview: '' }
      : {
          session_id: item.session_id,
          last_used: item.last_used || 0,
          message_count: item.message_count || 0,
          preview: item.preview || '',
        }
  )
}

// Prefer the first question as the human-readable session title.
function sessionTitle(session) {
  return truncate(session.preview) || truncate(session.session_id, 36) || 'Untitled session'
}

function formatRelativeTime(ts) {
  if (!ts) return ''
  const diffMs = Date.now() - ts * 1000
  const minutes = Math.floor(diffMs / 60000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}d ago`
  const date = new Date(ts * 1000)
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}



function HistoryPanel() {
  const [searchParams, setSearchParams] = useSearchParams()
  const urlSessionId = searchParams.get('session_id')

  const timerRef = useRef(null)
  const listRef = useRef(null)
  const stickToBottomRef = useRef(true)
  const streamingRef = useRef(false)
  const recognitionRef = useRef(null)
  const mockTimerRef = useRef(null)

  const [sessions, setSessions] = useState([])
  const [selectedSession, setSelectedSession] = useState(urlSessionId)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [shareFeedback, setShareFeedback] = useState('')
  const [copiedIndex, setCopiedIndex] = useState(null)
  const [feedbackMap, setFeedbackMap] = useState({}) // 'like', 'dislike', or null
  const [showThankYou, setShowThankYou] = useState(false)
  const [copyFeedback, setCopyFeedback] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [queryText, setQueryText] = useState('')
  const [sending, setSending] = useState(false)
  const [streamError, setStreamError] = useState('')
  const [isListening, setIsListening] = useState(false)
  const [speechSupported, setSpeechSupported] = useState(false)

  const searchTerm = searchQuery.toLowerCase()
  // Sessions in time order — most recently used first.
  const filteredSessions = sessions
    .filter(s => `${s.session_id} ${s.preview}`.toLowerCase().includes(searchTerm))
    .sort((a, b) => (b.last_used || 0) - (a.last_used || 0))

  const currentSession = sessions.find(s => s.session_id === selectedSession)

  // Parse "User: ...\nAssistant: ..." entries from the history array
  function parseHistory(historyList) {
    if (!Array.isArray(historyList)) return []
    return historyList.map(entry => {
      const lines = entry.split('\n')
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

  // Keep the selected session in sync with the URL (?session_id=...)
  useEffect(() => {
    if (urlSessionId) {
      setSelectedSession(urlSessionId)
    }
  }, [urlSessionId])

  // Select a session and mirror it into the URL so it is shareable.
  const selectSession = (id) => {
    setSelectedSession(id)
    setSearchParams({ session_id: id })
    setSidebarOpen(false)
  }

  // Handle sharing the conversation
  const handleShareExchange = async (query, response, index) => {
    const text = `You: ${query}\nAssistant: ${response}`

    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Share DevWhisper Conversation',
          text: text,
        })

        setCopiedIndex(index)
        setTimeout(() => setCopiedIndex(null), 2000)
        return
      } catch (err) {
        if (err.name === 'AbortError') {
          console.log('Share action was aborted')
        } else {
          console.error('Error sharing:', err)
          setCopiedIndex(null)
        }
      }
    }

    try {
      await navigator.clipboard.writeText(text)
      setCopiedIndex(index)
      setTimeout(() => setCopiedIndex(null), 2000)
    } catch (err) {
      console.warn('Error copying text:', err)
      const textarea = document.createElement('textarea')
      textarea.value = text
      document.body.appendChild(textarea)
      textarea.select()
      try {
        document.execCommand('copy')
        setCopiedIndex(index)
        setTimeout(() => setCopiedIndex(null), 2000)
      } catch (err) {
        console.error('Error copying text:', err)
      } finally {
        document.body.removeChild(textarea)
      }
    }
  }

  const handleShareSessionLink = async () => {
    if (!selectedSession) {
      setShareFeedback('Please select a session to share')
      setTimeout(() => setShareFeedback(''), 2000)
      return
    }
    const shareUrl = `${window.location.origin}/history?session_id=${encodeURIComponent(selectedSession)}`
    try {
      await navigator.clipboard.writeText(shareUrl)
      setShareFeedback('Share link copied to clipboard')
      setTimeout(() => setShareFeedback(''), 2000)
    } catch (err) {
      console.error('Error copying share link:', err)
      setShareFeedback('Failed to copy share link')
      setTimeout(() => setShareFeedback(''), 2000)
    }
  }

  const handleLike = (index) => {
    if (timerRef.current) clearTimeout(timerRef.current)
    setFeedbackMap(prev => {
      const current = prev[index]
      const newVal = current === 'like' ? null : 'like'
      return { ...prev, [index]: newVal }
    })
    setShowThankYou(true)
    timerRef.current = setTimeout(() => {
      setShowThankYou(false)
      timerRef.current = null
    }, 2500)
  }

  const handleDislike = (index) => {
    if (timerRef.current) clearTimeout(timerRef.current)
    setFeedbackMap(prev => {
      const current = prev[index]
      const newVal = current === 'dislike' ? null : 'dislike'
      return { ...prev, [index]: newVal }
    })
    setShowThankYou(true)
    timerRef.current = setTimeout(() => {
      setShowThankYou(false)
      timerRef.current = null
    }, 2500)
  }

  const handleCopy = async (response, index) => {
    // Try clipboard API first
    try {
      await navigator.clipboard.writeText(response)
      setCopyFeedback(index)
      setTimeout(() => setCopyFeedback(null), 2000)
      return
    } catch (err) {
      console.warn('Clipboard API failed, trying fallback:', err)
    }

    // Fallback: execCommand
    try {
      const textarea = document.createElement('textarea')
      textarea.value = response
      textarea.style.position = 'fixed'
      textarea.style.opacity = '0'
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
      setCopyFeedback(index)
      setTimeout(() => setCopyFeedback(null), 2000)
    } catch (err) {
      console.error('All copy methods failed:', err)
    }
  }

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [])

  // Poll session list every POLL_INTERVAL ms
  useEffect(() => {
    let active = true

    function fetchSessions() {
      fetch('/history')
        .then(res => res.json())
        .then(data => {
          if (!active) return
          const ids = data.session_ids || []
          const normalized = normalizeSessions(data.sessions || ids)
          setSessions(prev => {
            // Only update if the list actually changed
            if (prev.length === normalized.length &&
                prev.every((s, i) => s.session_id === normalized[i].session_id)) {
              return prev
            }
            return normalized
          })
          // Prefer the URL session, then the current one, then the newest.
          setSelectedSession(prev => {
            if (ids.length === 0) return null
            if (urlSessionId && ids.includes(urlSessionId)) return urlSessionId
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
  }, [urlSessionId])

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
          // Don't clobber the in-flight optimistic message while streaming.
          if (streamingRef.current) return
          const parsed = parseHistory(data.history || [])
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

  // Auto-scroll to the newest message when the message count or session changes
  useEffect(() => {
    const el = listRef.current
    if (el) {
      el.scrollTop = el.scrollHeight
    }
    stickToBottomRef.current = true
  }, [history.length, selectedSession])

  // Stop auto-sticking to the bottom if the user scrolls up to read history
  const handleScroll = () => {
    const el = listRef.current
    if (!el) return
    stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120
  }

  // Static /webhook fallback (mirrors the home page's query pipeline)
  const webhookFallback = async (q, sessionId) => {
    const payload = {
      message: {
        type: 'tool-calls',
        sessionId: sessionId,
        toolCalls: [
          {
            id: crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substring(2),
            type: 'function',
            function: {
              name: 'query_codebase',
              arguments: JSON.stringify({ query: q }),
            },
          },
        ],
      },
    }

    const res = await fetch('/webhook', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!res.ok) {
      throw new Error(`Server returned HTTP ${res.status}`)
    }

    const data = await res.json()
    if (data.status === 'error') {
      throw new Error(data.message || 'Error executing codebase query.')
    }

    const resultText = data.results?.[0]?.result || 'No response was generated by the codebase assistant.'
    setHistory(prev => {
      const copy = prev.slice()
      const last = copy[copy.length - 1]
      copy[copy.length - 1] = { ...last, response: resultText }
      return copy
    })
  }

  // Send a follow-up message from the history page and stream the reply.
  const submitQuery = async (q, sessionId) => {
    streamingRef.current = true
    stickToBottomRef.current = true
    setSending(true)

    try {
      const res = await fetch('/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, sessionId: sessionId }),
      })

      if (!res.ok) {
        if (res.status === 404) {
          await webhookFallback(q, sessionId)
          return
        }
        throw new Error(`Streaming failed: HTTP ${res.status}`)
      }

      if (!res.body) {
        throw new Error('Readable stream not supported or empty body returned.')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let accumulated = ''
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        accumulated += decoder.decode(value, { stream: !done })
        setHistory(prev => {
          const copy = prev.slice()
          const last = copy[copy.length - 1]
          copy[copy.length - 1] = { ...last, response: accumulated }
          return copy
        })
        if (stickToBottomRef.current && listRef.current) {
          listRef.current.scrollTop = listRef.current.scrollHeight
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return
      console.warn('Streaming failed, falling back to static webhook:', err)
      try {
        await webhookFallback(q, sessionId)
      } catch (fallbackErr) {
        console.error('Webhook fallback failed:', fallbackErr)
        setStreamError(fallbackErr.message || 'Failed to reach the DevWhisper backend.')
        // Drop the optimistic entry so polling can reconcile with the backend.
        setHistory(prev => prev.slice(0, -1))
      }
    } finally {
      streamingRef.current = false
      setSending(false)
    }
  }

  const handleSubmit = (e) => {
    if (e) e.preventDefault()
    if (!selectedSession || sending) return
    const q = queryText.trim()
    if (!q) return
    setQueryText('')
    setStreamError('')
    // Optimistically show the user's question right away.
    setHistory(prev => [...prev, { query: q, response: '' }])
    submitQuery(q, selectedSession)
  }

  // Initialize Speech Recognition API (mirrors the home page's voice flow)
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (SpeechRecognition) {
      setSpeechSupported(true)
      const rec = new SpeechRecognition()
      rec.continuous = false // Auto-stop when user stops speaking
      rec.interimResults = true
      rec.lang = 'en-US'

      rec.onstart = () => {
        setIsListening(true)
        setStreamError('')
      }

      rec.onresult = (event) => {
        let transcript = ''
        for (let i = 0; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript
        }
        setQueryText(transcript)
      }

      rec.onerror = (err) => {
        console.error('Speech Recognition Error:', err)
        setIsListening(false)
      }

      rec.onend = () => {
        // Keep the recognized transcript in the input box — the user reviews
        // it and presses Send manually. No auto-submit.
        setIsListening(false)
      }

      recognitionRef.current = rec
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort()
      }
      if (mockTimerRef.current) {
        clearTimeout(mockTimerRef.current)
        mockTimerRef.current = null
      }
    }
  }, [])

  const handleMicClick = () => {
    if (speechSupported && recognitionRef.current) {
      if (isListening) {
        recognitionRef.current.stop()
      } else {
        setQueryText('')
        setStreamError('')
        try {
          recognitionRef.current.start()
        } catch (err) {
          console.error('Failed to start speech recognition:', err)
          setStreamError('Failed to activate microphone. Please try again.')
        }
      }
    } else {
      // Mock fallback for browsers/environments without SpeechRecognition
      if (isListening) {
        setIsListening(false)
        if (mockTimerRef.current) {
          clearTimeout(mockTimerRef.current)
          mockTimerRef.current = null
        }
      } else {
        setIsListening(true)
        setQueryText('Listening...')
        setStreamError('')
        if (mockTimerRef.current) {
          clearTimeout(mockTimerRef.current)
        }
        mockTimerRef.current = setTimeout(() => {
          setIsListening(false)
          // Mock result — fill the input box only, no auto-submit.
          setQueryText('In main.py, what functions are found?')
          mockTimerRef.current = null
        }, 3000)
      }
    }
  }

  if (loading) {
    return (
      <div className="history-panel">
        <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
          <div className="sidebar-header">
            <h3>Sessions</h3>
            <button className="sidebar-close" onClick={() => setSidebarOpen(false)}>✕</button>
          </div>
          <nav className="sidebar-list">
            <p className="sidebar-empty">Loading sessions...</p>
          </nav>
        </aside>
        <header className="chat-header">
          <button className="hamburger" onClick={() => setSidebarOpen(true)} aria-label="Open sessions">≡</button>
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
      {/* Sidebar overlay */}
      {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />}

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
        <div className="sidebar-header">
          <h3>Sessions</h3>
          <button className="sidebar-close" onClick={() => setSidebarOpen(false)}>✕</button>
        </div>
        <div className="sidebar-search">
          <input
            type="text"
            className="search-input"
            placeholder="Search sessions..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
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
        <nav className="sidebar-list">
          {filteredSessions.length === 0 ? (
            <p className="sidebar-empty">
              {sessions.length === 0 ? "No sessions yet" : "No matching sessions"}
            </p>
          ) : (
            filteredSessions.map(session => (
              <button
                key={session.session_id}
                className={`sidebar-item ${session.session_id === selectedSession ? 'sidebar-item--active' : ''}`}
                onClick={() => selectSession(session.session_id)}
              >
                <span className="sidebar-item-title">
                  {highlightText(sessionTitle(session), searchQuery)}
                </span>
                <span className="sidebar-item-meta">
                  {formatRelativeTime(session.last_used)}
                  {session.message_count > 0 && ` · ${session.message_count} msg${session.message_count > 1 ? 's' : ''}`}
                </span>
              </button>
            ))
          )}
        </nav>
      </aside>

      {/* Header */}
      <header className="chat-header">
        <button className="hamburger" onClick={() => setSidebarOpen(true)} aria-label="Open sessions">≡</button>
        <div className="chat-header-info">
          <h2>History</h2>
          <p className="subtitle">{currentSession ? sessionTitle(currentSession) : 'Conversation history'}</p>
        </div>
        <div className="header-actions">
          <button className="share-button" onClick={handleShareSessionLink}>
            🔗 Share link
          </button>
          {shareFeedback && <span className="share-feedback">{shareFeedback}</span>}
          <Link to="/" className="back-link">← Home</Link>
        </div>
      </header>

      {/* Current session summary */}
      {currentSession && (
        <div className="session-detail">
          <span className="session-detail-id" title={currentSession.session_id}>
            {currentSession.session_id}
          </span>
          {currentSession.last_used > 0 && (
            <span className="session-detail-time">
              Last used {formatRelativeTime(currentSession.last_used)}
            </span>
          )}
          {currentSession.message_count > 0 && (
            <span className="session-detail-count">
              {currentSession.message_count} message{currentSession.message_count > 1 ? 's' : ''}
            </span>
          )}
        </div>
      )}

      {/* Scrollable conversation */}
      <div className="chat-messages" ref={listRef} onScroll={handleScroll}>
        {history.length === 0 ? (
          <div className="history-empty">
            <div className="empty-icon">💬</div>
            <h3>No messages yet</h3>
            <p>Ask something below to start the conversation</p>
          </div>
        ) : (
          history.map((item, index) => (
            <div key={index} className="exchange">
              <div className="chat-message chat-message--user">
                <div className="chat-bubble chat-bubble--user">{item.query}</div>
              </div>
              <div className="chat-message chat-message--assistant">
                <div className="chat-bubble chat-bubble--assistant response-body">
                  <Markdown text={item.response} />
                </div>
                <div className="chat-actions">
                  <button
                    className="share-exchange-btn"
                    onClick={() => handleShareExchange(item.query, item.response, index)}
                    aria-label="Share this exchange">
                    {copiedIndex === index ? 'Copied!' : 'Share'}
                  </button>
                  <button
                    className={`like-button ${feedbackMap[index] === 'like' ? 'active' : ''}`}
                    onClick={() => handleLike(index)}
                  >
                    👍
                  </button>
                  <button
                    className={`dislike-button ${feedbackMap[index] === 'dislike' ? 'active' : ''}`}
                    onClick={() => handleDislike(index)}
                  >
                    👎
                  </button>
                  <button
                    className={`copy-button ${copyFeedback === index ? 'active' : ''}`}
                    onClick={() => handleCopy(item.response, index)}
                  >
                    {copyFeedback === index ? '✔' : '📋︎'}
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {streamError && <div className="stream-error">⚠️ {streamError}</div>}

      {/* Pinned input bar — always stays at the bottom */}
      <form className="chat-input-bar" onSubmit={handleSubmit}>
        <MicButton isListening={isListening} onClick={handleMicClick} disabled={sending} />
        <textarea
          className="chat-input"
          value={queryText}
          placeholder={isListening
            ? "Listening... Speak now."
            : selectedSession
              ? "Ask a follow-up..."
              : "Select a session from the sidebar to start chatting"}
          disabled={sending}
          rows={1}
          onChange={e => {
            setQueryText(e.target.value)
            const el = e.target
            el.style.height = 'auto'
            el.style.height = Math.min(el.scrollHeight, 160) + 'px'
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleSubmit()
            }
          }}
        />
        <button type="submit" className="send-button" disabled={sending || !queryText.trim() || !selectedSession}>
          {sending ? 'Sending…' : 'Send'}
        </button>
      </form>

      {showThankYou && (
        <div className="thank-you-toast">Thanks for your feedback 🙌</div>
      )}
    </div>
  )
}

export default HistoryPanel
