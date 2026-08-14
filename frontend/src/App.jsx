import { useState, useEffect, useRef, useCallback } from 'react'
import { BrowserRouter, Routes, Route, Link, useNavigate } from 'react-router-dom'
import HistoryPanel from './components/HistoryPanel.jsx'
import ResponseOutput from './components/ResponseOutput.jsx'
import MicButton from './components/MicButton.jsx'
import ProcessingTimeline from './components/ProcessingTimeline.jsx'
import './App.css'
import SettingsPanel from './components/SettingsPanel.jsx'

function Home() {
  const [queryText, setQueryText] = useState('')
  const [response, setResponse] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isListening, setIsListening] = useState(false)
  const [speechSupported, setSpeechSupported] = useState(false)
  const [countdown, setCountdown] = useState(null)
  const [lastSubmittedQuery, setLastSubmittedQuery] = useState('')
  const [reindexRecommended, setReindexRecommended] = useState(false)
  const [suggestions, setSuggestions] = useState([])

  const recognitionRef = useRef(null)
  const isMountedRef = useRef(false)
  const abortControllerRef = useRef(null)
  const mockTimerRef = useRef(null)
  const latestTranscriptRef = useRef('')
  const countdownIntervalRef = useRef(null)

  // Retrieve or generate a stable session ID so query history shows up in history panel
  const [hasStarted, setHasStarted] = useState(false)

  const SUGGESTED_PROMPTS = [
    "What does the preprocess function do?",
    "Where is the model saved after training?",
    "How do I debug a KeyError in the pipeline?",
    "What functions are defined in main.py?"
  ]

  const submitQueryTextRef = useRef(null)
  const redirectedRef = useRef(false)

  const navigate = useNavigate()

  // Retrieve or generate a stable session ID so query history shows up in history panel
  const [sessionId, setSessionId] = useState(() => {
    const key = 'devwhisper_session_id'
    const existing = sessionStorage.getItem(key)
    if (existing) return existing
    const newId = 'web-' + Math.random().toString(36).substring(2, 9)
    sessionStorage.setItem(key, newId)
    return newId
  })

  const getRecordingTimeout = () =>
    parseInt(localStorage.getItem('devwhisper_recording_timeout') || '30', 10)

  const stopCountdown = useCallback(() => {
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current)
      countdownIntervalRef.current = null
    }
    setCountdown(null)
  }, [])

  const startCountdown = useCallback((seconds, onExpire) => {
    setCountdown(seconds)
    let remaining = seconds
    countdownIntervalRef.current = setInterval(() => {
      remaining -= 1
      if (isMountedRef.current) setCountdown(remaining)
      if (remaining <= 0) {
        clearInterval(countdownIntervalRef.current)
        countdownIntervalRef.current = null
        onExpire()
      }
    }, 1000)
  }, [])

  const handleTextareaInput = (e) => {
    e.target.style.height = 'auto'
    e.target.style.height = `${Math.min(e.target.scrollHeight, 280)}px`
  }

  // Clear conversation handler
  const handleClearChat = async () => {
    if (loading) return

    setResponse('')
    setQueryText('')
    setError(null)
    setHasStarted(false)

    const newId = 'web-' + Math.random().toString(36).substring(2, 9)
    sessionStorage.setItem('devwhisper_session_id', newId)
    setSessionId(newId)
    // Allow the fresh session to get its own hand-off to /history.
    redirectedRef.current = false

    try {
      const res = await fetch('/reset', {
        method: 'POST'
      })
      if (!res.ok) {
        console.error('Failed to reset conversation memory.')
      }
      const suggestionsRes = await fetch('/index/suggestions')
      if (suggestionsRes.ok) {
        const data = await suggestionsRes.json()
        setSuggestions(data.suggestions || [])
      }
    } catch (err) {
      console.error('Error resetting conversation memory:', err)
    }
  }

  // Webhook Fallback Function
  const handleWebhookFallback = useCallback(async (fallbackQuery) => {
    const q = fallbackQuery !== undefined ? fallbackQuery : queryText
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
              arguments: JSON.stringify({ query: q })
            }
          }
        ]
      }
    }

    const res = await fetch('/webhook', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    })

    if (!res.ok) {
      throw new Error(`Server returned HTTP ${res.status}`)
    }

    const data = await res.json()
    if (data.status === 'error') {
      throw new Error(data.message || 'Error executing codebase query.')
    }

    const resultText = data.results?.[0]?.result
    if (isMountedRef.current) {
      if (resultText) {
        setResponse(resultText)
      } else {
        setResponse('No response was generated by the codebase assistant.')
      }
    }
  }, [queryText, sessionId])

  // Timeline stages state for Issue #221
  const INITIAL_STAGES = [
    { id: 'request_received', label: 'Request Received', status: 'pending' },
    { id: 'retrieval', label: 'Code Context Retrieval', status: 'pending' },
    { id: 'generation', label: 'LLM Generation', status: 'pending' },
    { id: 'completion', label: 'Response Completed', status: 'pending' },
  ]
  const [timelineStages, setTimelineStages] = useState([])

  const updateStage = (stageId, status, detail = '') => {
    if (!isMountedRef.current) return
    setTimelineStages((prev) =>
      prev.map((st) => (st.id === stageId ? { ...st, status, detail } : st))
    )
  }

  // Submit Query Function
  const submitQueryText = useCallback(async (textToSubmit) => {
    const currentQuery = textToSubmit || queryText
    if (!currentQuery.trim() || loading) return

    setLastSubmittedQuery(currentQuery)

    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    abortControllerRef.current = new AbortController()
    const signal = abortControllerRef.current.signal

    setHasStarted(true)
    setLoading(true)
    setError(null)
    setResponse('')
    setTimelineStages([
      { id: 'request_received', label: 'Request Received', status: 'completed', detail: 'Query parsed' },
      { id: 'retrieval', label: 'Code Context Retrieval', status: 'in_progress', detail: 'Searching codebase...' },
      { id: 'generation', label: 'LLM Generation', status: 'pending' },
      { id: 'completion', label: 'Response Completed', status: 'pending' },
    ])

    try {
      const responseStream = await fetch('/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: currentQuery, sessionId: sessionId }),
        signal
      })

      if (!responseStream.ok) {
        if (responseStream.status === 404) {
          await handleWebhookFallback(currentQuery)
          return
        }
        throw new Error(`Streaming failed: HTTP ${responseStream.status}`)
      }

      if (!responseStream.body) {
        throw new Error('Readable stream not supported or empty body returned.')
      }

      setTimelineStages([
        { id: 'request_received', label: 'Request Received', status: 'completed' },
        { id: 'retrieval', label: 'Code Context Retrieval', status: 'completed', detail: 'Context retrieved' },
        { id: 'generation', label: 'LLM Generation', status: 'in_progress', detail: 'Streaming tokens...' },
        { id: 'completion', label: 'Response Completed', status: 'pending' },
      ])

      const reader = responseStream.body.getReader()
      const decoder = new TextDecoder()
      let done = false
      let accumulated = ''

      while (!done) {
        const { value, done: readerDone } = await reader.read()
        done = readerDone
        if (value) {
          const chunk = decoder.decode(value, { stream: !done })
          accumulated += chunk
          if (isMountedRef.current) {
            setResponse(accumulated)
          }
        }
      }

      setTimelineStages([
        { id: 'request_received', label: 'Request Received', status: 'completed' },
        { id: 'retrieval', label: 'Code Context Retrieval', status: 'completed' },
        { id: 'generation', label: 'LLM Generation', status: 'completed' },
        { id: 'completion', label: 'Response Completed', status: 'completed', detail: 'Finished' },
      ])

    } catch (err) {
      if (err.name === 'AbortError') {
        return
      }
      console.warn('Streaming failed or not available, falling back to static webhook:', err)
      if (isMountedRef.current) {
        try {
          await handleWebhookFallback(currentQuery)
          setTimelineStages([
            { id: 'request_received', label: 'Request Received', status: 'completed' },
            { id: 'retrieval', label: 'Code Context Retrieval', status: 'completed' },
            { id: 'generation', label: 'LLM Generation', status: 'completed' },
            { id: 'completion', label: 'Response Completed', status: 'completed' },
          ])
        } catch (fallbackErr) {
          if (isMountedRef.current) {
            setError(fallbackErr.message || 'Failed to reach the DevWhisper backend.')
            setTimelineStages((prev) =>
              prev.map((st) => (st.status === 'in_progress' ? { ...st, status: 'failed' } : st))
            )
          }
        }
      }
    } finally {
      if (isMountedRef.current) {
        setLoading(false)
      }
    }
  }, [queryText, loading, sessionId, handleWebhookFallback])

  // Form Submit Handler
  const handleSubmit = (e) => {
    if (e) e.preventDefault()
    if (isListening) return
    submitQueryText(queryText)
  }

  const handleRetry = useCallback(() => {
    submitQueryText(lastSubmittedQuery)
  }, [submitQueryText, lastSubmittedQuery])

  const checkReindex = useCallback(async () => {
    try {
      const res = await fetch('/index/change', { method: 'GET' })
      if (res.ok) {
        const data = await res.json()
        setReindexRecommended(data.reindex_recommended)
      }
    }
    catch (err) {
      console.error('Error checking reindex recommendation:', err)
    }
  }, [])

  useEffect(() => {
    checkReindex()
    const timer = setInterval(checkReindex, 30000)
    return () => clearInterval(timer)
  }, [checkReindex])

  useEffect(() => {
    // Refresh the banner right away when the active repository changes
    window.addEventListener('repo-changed', checkReindex)
    return () => window.removeEventListener('repo-changed', checkReindex)
  }, [checkReindex])

  useEffect(() => {
    let active = true
    const fetchSuggestions = async () => {
      try {
        const res = await fetch('/index/suggestions')
        if (res.ok && active) {
          const data = await res.json()
          setSuggestions(data.suggestions || [])
        }
      } catch (err) {
        console.error('Failed to load query suggestions:', err)
      }
    }

    fetchSuggestions()
    const timer = setInterval(fetchSuggestions, 15000)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    let active = true
    const fetchSuggestions = async () => {
      try {
        const res = await fetch('/index/suggestions')
        if (res.ok && active) {
          const data = await res.json()
          setSuggestions(data.suggestions || [])
        }
      } catch (err) {
        console.error('Failed to load query suggestions:', err)
      }
    }

    fetchSuggestions()
    const timer = setInterval(fetchSuggestions, 15000)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [])

  // Keep a ref pointing at the latest submitQueryText so the mount-only
  // speech-recognition effect never goes stale
  useEffect(() => {
    submitQueryTextRef.current = submitQueryText
  }, [submitQueryText])

  // After the first exchange completes, hand off to the conversation view.
  useEffect(() => {
    if (!loading && response && !redirectedRef.current) {
      redirectedRef.current = true
      navigate(`/history?session_id=${encodeURIComponent(sessionId)}`)
    }
  }, [loading, response, sessionId, navigate])

  // Initialize Speech Recognition API with complete E2E Voice Flow
  useEffect(() => {
    isMountedRef.current = true
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (SpeechRecognition) {
      setSpeechSupported(true)
      const rec = new SpeechRecognition()
      rec.continuous = false // Auto-stop when user stops speaking
      rec.interimResults = true
      rec.lang = 'en-US'

      rec.onstart = () => {
        if (isMountedRef.current) {
          setIsListening(true)
          setError(null)
        }
      }

      rec.onresult = (event) => {
        let transcript = ''
        for (let i = 0; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript
        }
        if (isMountedRef.current) {
          setQueryText(transcript)
          latestTranscriptRef.current = transcript
        }
      }

      rec.onerror = (err) => {
        console.error('Speech Recognition Error:', err)
        if (isMountedRef.current) {
          stopCountdown()
          setIsListening(false)
          if (err.error === 'not-allowed' || err.error === 'service-not-allowed') {
            setError('Microphone permission denied. Please allow mic access in your browser.')
          } else if (err.error === 'no-speech') {
            setError('No speech was detected. Please try pressing the mic and speaking again.')
          } else if (err.error === 'network') {
            setError('Speech recognition failed due to a network error.')
          } else {
            setError(`Speech recognition error: ${err.error}`)
          }
        }
      }

      rec.onend = () => {
        if (isMountedRef.current) {
          stopCountdown()
          setIsListening(false)
          const finalQuery = latestTranscriptRef.current.trim()
          if (finalQuery) {
            submitQueryTextRef.current(finalQuery)
          }
        }
      }

      recognitionRef.current = rec
    }

    return () => {
      isMountedRef.current = false

      if (recognitionRef.current) {
        recognitionRef.current.abort()
      }

      stopCountdown()

      if (mockTimerRef.current) {
        clearTimeout(mockTimerRef.current)
        mockTimerRef.current = null
      }
    }
  }, [stopCountdown])

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

  const handleMicClick = () => {
    if (speechSupported && recognitionRef.current) {
      if (isListening) {
        stopCountdown()
        recognitionRef.current.stop()
      } else {
        setQueryText('')
        latestTranscriptRef.current = ''
        setError(null)
        try {
          recognitionRef.current.start()
          startCountdown(getRecordingTimeout(), () => {
            if (recognitionRef.current) recognitionRef.current.stop()
          })
        } catch (err) {
          console.error('Failed to start speech recognition:', err)
          setError('Failed to activate microphone. Please try again.')
        }
      }
    } else {
      // Mock Fallback for browsers/environments without SpeechRecognition
      if (isListening) {
        setIsListening(false)
        stopCountdown()
        if (mockTimerRef.current) {
          clearTimeout(mockTimerRef.current)
          mockTimerRef.current = null
        }
      } else {
        setIsListening(true)
        setQueryText('Listening...')
        setError(null)
        const timeout = getRecordingTimeout()
        startCountdown(timeout, () => {
          if (isMountedRef.current) {
            setIsListening(false)
            const sampleQuery = 'In main.py, what functions are found?'
            setQueryText(sampleQuery)
            submitQueryText(sampleQuery)
          }
        })
        if (mockTimerRef.current) clearTimeout(mockTimerRef.current)
        mockTimerRef.current = setTimeout(() => {
          if (isMountedRef.current) {
            setIsListening(false)
            stopCountdown()
            const sampleQuery = 'In main.py, what functions are found?'
            setQueryText(sampleQuery)
            submitQueryText(sampleQuery)
            mockTimerRef.current = null
          }
        }, timeout * 1000)
      }
    }
  }

  return (
    <div className="landing-container">
      <header className="hero-header">
        <h1 className="logo-text">DevWhisper</h1>
        <p className="subtitle-text">Voice-native developer experience agent</p>
      </header>

      {reindexRecommended && (
        <div className="reindex-banner">
          ⚠️ Codebase changed. Re-indexing is recommended.
        </div>
      )}

      <main className="query-card">
        <form onSubmit={handleSubmit} className="query-form">
          <div className="textarea-wrapper">
            <textarea
              value={queryText}
              onChange={e => setQueryText(e.target.value)}
              onInput={handleTextareaInput}
              placeholder={isListening ? "Listening... Speak now." : "Ask about your codebase... (e.g., In main.py, what functions are found?)"}
              disabled={loading}
              rows={3}
              className="query-input"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSubmit()
                }
              }}
            />

            <div className="query-toolbar">
              <div className="toolbar-left">
                <MicButton
                  isListening={isListening}
                  onClick={handleMicClick}
                  disabled={loading}
                  countdown={countdown}
                />
                <div className="voice-status-info">
                  <span className={`status-dot ${isListening ? 'listening' : 'ready'}`}></span>
                  <span className="status-message">
                    {isListening
                      ? (speechSupported ? "Listening... Speak now" : "Listening (Mock)...")
                      : "Voice assistant ready"}
                  </span>
                </div>
              </div>

              <div className="toolbar-right">
                <button
                  type="button"
                  onClick={handleClearChat}
                  disabled={loading || (!queryText.trim() && !response && !error)}
                  className="clear-button"
                  title="Clear conversation"
                >
                  Clear Chat
                </button>
                <button
                  type="submit"
                  disabled={loading || !queryText.trim() || isListening}
                  className="submit-button"
                >
                  {loading ? 'Analyzing...' : 'Send Query'}
                </button>
              </div>
            </div>
          </div>
        </form>

        {/* Contextual suggestions */}
        {!response && !loading && !error && suggestions.length > 0 && (
          <div className="suggestions-container">
            <h3 className="suggestions-title">💡 Try asking:</h3>
            <div className="suggestions-grid">
              {suggestions.map((suggestion, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => {
                    setQueryText(suggestion)
                    submitQueryText(suggestion)
                  }}
                  className="suggestion-tag"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Processing timeline for intermediate progress */}
        <ProcessingTimeline stages={timelineStages} />

        {/* Response Rendering */}
        <ResponseOutput response={response} loading={loading} error={error} />
      </main>

      <footer className="landing-footer">
        <p className="session-info">
          Active Session ID: <code>{sessionId}</code>
        </p>
        <Link to="/history" className="history-link">
          📋 View Full Session History &rarr;
        </Link>
      </footer>
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <SettingsPanel />

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/history" element={<HistoryPanel />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App
