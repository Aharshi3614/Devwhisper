import { useState, useEffect, useCallback, use } from 'react'
import ThemeToggle from './ThemeToggle.jsx'
import './SettingsPanel.css'

export default function SettingsPanel() {
  const [isOpen, setIsOpen] = useState(false)
  
  // Future setting state placeholders (stored in localStorage for extension)
  const [autoSubmitVoice, setAutoSubmitVoice] = useState(() => {
    return localStorage.getItem('devwhisper_auto_submit') !== 'false'
  })

  const [recordingTimeout, setRecordingTimeout] = useState(() => {
    return parseInt(localStorage.getItem('devwhisper_recording_timeout') || '30', 10)
  })
  const [progress, setProgress] = useState(null)
  const [queue, setQueue] = useState([])
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [uploadSuccess, setUploadSuccess] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const [repos, setRepos] = useState([])
  const [currentRepoId, setCurrentRepoId] = useState(null)
  const [repoPath, setRepoPath] = useState('')

  useEffect(() => {
    localStorage.setItem('devwhisper_auto_submit', autoSubmitVoice)
  }, [autoSubmitVoice])

  useEffect(() => {
    localStorage.setItem('devwhisper_recording_timeout', recordingTimeout)
  }, [recordingTimeout])

  // Close modal on Escape key press
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen])

  // Listen to /index/progress SSE when modal is open
  useEffect(() => {
    if (!isOpen) return

    const eventSource = new EventSource('/index/progress')

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        setProgress(data)
      } catch (err) {
        console.error('Failed to parse indexing progress SSE data:', err)
      }
    }

    eventSource.onerror = (err) => {
      console.error('EventSource connection error:', err)
    }

    return () => {
      eventSource.close()
    }
  }, [isOpen])

  const fetchQueue = useCallback(async () => {
    try {
      const res = await fetch('/index/queue')
      if (res.ok) {
        const data = await res.json()
        setQueue(data.jobs || [])
      }
    } catch (err) {
      console.error('Failed to fetch indexing queue:', err)
    }
  }, [])

  const fetchRepos = useCallback(async () => {
    try {
      const res = await fetch('/repos')
      if (res.ok) {
        const data = await res.json()
        setRepos(data.repos || [])
        setCurrentRepoId(data.current || null)
      }
    } catch (err) {
      console.error('Failed to fetch repositories:', err)
    }
  }, [])

  useEffect(() => {
    if (!isOpen) return
    fetchRepos()
    const timer = setInterval(fetchRepos, 3000)
    return () => clearInterval(timer)
  }, [isOpen, fetchRepos])

  useEffect(() => {
    if (!isOpen) return
    fetchQueue()
    const timer = setInterval(fetchQueue, 3000)
    return () => clearInterval(timer)
  }, [isOpen, fetchQueue])

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const uploadFile = useCallback(async (file) => {
    if (!file.name.endsWith('.zip')) {
      setUploadError('Only .zip archives are supported.')
      setUploadSuccess('')
      return
    }

    setUploading(true)
    setUploadError('')
    setUploadSuccess('')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/index/upload', {
        method: 'POST',
        body: formData,
      })

      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.message || 'Upload failed.')
      }

      setUploadSuccess(data.message || 'ZIP uploaded successfully.')
      setUploadError('')
      fetchQueue()
    } catch (err) {
      setUploadError(err.message || 'An error occurred during upload.')
      setUploadSuccess('')
    } finally {
      setUploading(false)
    }
  }, [fetchQueue])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (uploading || (progress && progress.running)) return

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0])
    }
  }, [uploading, progress, uploadFile])

  const handleFileChange = useCallback((e) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      uploadFile(e.target.files[0])
    }
  }, [uploadFile])

  const handleManualIndex = useCallback(async () => {
    if (progress && progress.running) return
    setUploadError('')
    setUploadSuccess('')
    try {
      const res = await fetch('/index/start', { method: 'POST' })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.message || 'Failed to start indexing.')
      }
      setUploadSuccess(data.message || 'Indexing started.')
      fetchQueue()
    } catch (err) {
      setUploadError(err.message || 'Failed to start indexing.')
    }
  }, [progress, fetchQueue])

  const handleAddRepo = useCallback(async () => {
    if (!repoPath.trim()) return
    const res = await fetch('/repos/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: repoPath})
    })
    if (res.ok) {
      setRepoPath('')
      fetchRepos()
    } else {
      const data = await res.json()
      console.error('Failed to add repository:', data.message || res.status)
    }
  }, [repoPath, fetchRepos])

  const handleSwitchRepo = useCallback(async (id) => {
    const res = await fetch('/repos/switch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id: id })
    })

    if (res.ok) {
      fetchRepos()
      window.dispatchEvent(new Event('repo-changed')) 
    }
  }, [fetchRepos])

  return (
    <>
      {/* Settings Trigger Button */}
      <button
        type="button"
        className="settings-toggle-btn"
        onClick={() => setIsOpen(true)}
        aria-label="Open Settings"
        title="Settings & Preferences"
      >
        ⚙️
      </button>

      {/* Slide-out Panel or Modal */}
      {isOpen && (
        <div className="settings-overlay" onClick={() => setIsOpen(false)}>
          <div
            className="settings-modal"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-title"
          >
            <div className="settings-header">
              <h2 id="settings-title" className="settings-title">Preferences</h2>
              <button
                type="button"
                className="settings-close-btn"
                onClick={() => setIsOpen(false)}
                aria-label="Close Settings"
              >
                ✕
              </button>
            </div>

            <div className="settings-body">
              {/* Theme Settings Section */}
              <div className="settings-section">
                <div className="setting-item">
                  <div className="setting-info">
                    <span className="setting-label">Appearance Theme</span>
                    <span className="setting-desc">Toggle between Light and Dark mode</span>
                  </div>
                  <div className="setting-control">
                    <ThemeToggle />
                  </div>
                </div>
              </div>

              {/* Voice & Input Preferences Section */}
              <div className="settings-section">
                <div className="setting-item">
                  <div className="setting-info">
                    <span className="setting-label">Auto-submit Voice Queries</span>
                    <span className="setting-desc">Automatically send queries after speech ends</span>
                  </div>
                  <div className="setting-control">
                    <input
                      type="checkbox"
                      id="auto-submit-toggle"
                      checked={autoSubmitVoice}
                      onChange={(e) => setAutoSubmitVoice(e.target.checked)}
                      className="setting-checkbox"
                    />
                  </div>
                </div>
                <div className="setting-item" style={{ marginTop: '12px' }}>
                  <div className="setting-info">
                    <span className="setting-label">Max Recording Timeout</span>
                    <span className="setting-desc">Auto-stop recording after this many seconds (5–120)</span>
                  </div>
                  <div className="setting-control">
                    <input
                      type="number"
                      min="5"
                      max="120"
                      value={recordingTimeout}
                      onChange={(e) => setRecordingTimeout(Math.min(120, Math.max(5, Number(e.target.value))))}
                      className="setting-number-input"
                      aria-label="Max recording timeout in seconds"
                    />
                    <span style={{ fontSize: '0.8rem', color: '#666', marginLeft: '4px' }}>s</span>
                  </div>
                </div>
              </div>

              {/* Codebase Management Section */}
              <div className="settings-section">
                <div className="setting-item vertical">
                  <span className="setting-label">Manage Local Codebase</span>
                  <span className="setting-desc">Upload a ZIP file of your codebase to index and query.</span>

                  <div 
                    className={`upload-zone ${dragActive ? 'active' : ''} ${uploading || (progress && progress.running) ? 'disabled' : ''}`}
                    onDragEnter={handleDrag}
                    onDragOver={handleDrag}
                    onDragLeave={handleDrag}
                    onDrop={handleDrop}
                  >
                    <input
                      type="file"
                      id="zip-upload-input"
                      accept=".zip"
                      onChange={handleFileChange}
                      disabled={uploading || (progress && progress.running)}
                      className="upload-input-hidden"
                    />
                    <label htmlFor="zip-upload-input" className="upload-label">
                      <span className="upload-icon">📁</span>
                      {uploading ? (
                        <span>Uploading codebase ZIP...</span>
                      ) : progress && progress.running ? (
                        <span>Indexing in progress...</span>
                      ) : (
                        <span>Drag & drop ZIP here, or <strong>click to browse</strong></span>
                      )}
                    </label>
                  </div>

                  {uploadError && <div className="upload-feedback error">{uploadError}</div>}
                  {uploadSuccess && <div className="upload-feedback success">{uploadSuccess}</div>}

                  {/* Manual Index Trigger */}
                  {!uploading && (!progress || !progress.running) && (
                    <button
                      type="button"
                      onClick={handleManualIndex}
                      className="manual-index-btn"
                    >
                      🔄 Re-index Current Codebase
                    </button>
                  )}

                  {/* Indexing Progress Indicator */}
                  {progress && (progress.running || progress.status === 'done' || progress.status === 'error') && (
                    <div className="indexing-progress-container">
                      <div className="progress-status-header">
                        <span className="progress-status-title">
                          Status: <strong className={`status-${progress.status}`}>{progress.status.toUpperCase()}</strong>
                        </span>
                        {progress.running && (
                          <span className="progress-percentage">{progress.percent}%</span>
                        )}
                      </div>

                      {progress.running && (
                        <div className="progress-bar-wrapper">
                          <div 
                            className="progress-bar-fill" 
                            style={{ width: `${progress.percent}%` }}
                          />
                        </div>
                      )}

                      <div className="progress-status-msg">{progress.message}</div>
                      {progress.current_file && (
                        <div className="progress-current-file">
                          Processing: <code>{progress.current_file}</code>
                        </div>
                      )}
                      {progress.skipped_count > 0 && (
                        <div className="progress-skipped-info">
                          ⚠️ {progress.skipped_count} file(s) skipped.
                        </div>
                      )}
                      {progress.circular_imports && progress.circular_imports.length > 0 && (
                        <div className="progress-circular-info">
                          🔄 Circular imports detected: {progress.circular_imports.join('; ')}
                        </div>
                      )}
                      {progress.chunk_statistics && progress.chunk_statistics.total_chunks > 0 && (
                        <div className="progress-chunk-stats">
                          📊 <strong>{progress.chunk_statistics.total_chunks}</strong> chunks · avg{' '}
                          <strong>{progress.chunk_statistics.average_size}</strong> lines
                          {progress.chunk_statistics.largest && (
                            <span className="chunk-stat-detail">
                              {' '}· largest {progress.chunk_statistics.largest.size} lines ({progress.chunk_statistics.largest.file}:{progress.chunk_statistics.largest.start_line}) ·{' '}
                              smallest {progress.chunk_statistics.smallest.size} lines ({progress.chunk_statistics.smallest.file}:{progress.chunk_statistics.smallest.start_line})
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Indexing Queue List */}
                  {queue.length > 0 && (
                    <div className="indexing-queue-container">
                      <h4 className="queue-title">📋 Queue Jobs ({queue.filter(j => j.status === 'pending' || j.status === 'running').length} active)</h4>
                      <div className="queue-list">
                        {queue.map((job) => (
                          <div key={job.id} className={`queue-item status-${job.status}`}>
                            <div className="queue-item-header">
                              <span className="job-name">{job.name}</span>
                              <span className={`job-status-tag status-${job.status}`}>
                                {job.status.toUpperCase()}
                              </span>
                            </div>
                            <div className="queue-item-meta">
                              {job.status === 'running' && progress && (
                                <span className="job-progress-pct">{progress.percent}%</span>
                              )}
                              {job.status === 'failed' && (
                                <span className="job-error">⚠️ {job.error}</span>
                              )}
                              {job.status === 'completed' && (
                                <span className="job-success">✓ Done</span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
                  
              {/* Repositories Section */}
              <div className="settings-section">
                <div className="setting-item vertical">
                  <span className="setting-label">Repositories</span>
                  <span className="setting-desc">Add server paths and switch the active repository.</span>

                  <div className="repo-add-row">
                    <input
                      type="text"
                      placeholder="Server path, e.g. C:/projects/foo"
                      value={repoPath}
                      onChange={(e) => setRepoPath(e.target.value)}
                      className="repo-path-input"
                    />
                    <button type="button" onClick={handleAddRepo} className="repo-add-btn">➕ Add</button>
                  </div>

                  <div className="repo-list">
                    {repos.map((repo) => (
                      <div
                        key={repo.id}
                        className={`repo-item ${currentRepoId === repo.id ? 'active' : ''}`}
                        onClick={() => handleSwitchRepo(repo.id)}
                        role="button"
                        tabIndex={0}
                      >
                        <span className="repo-name">📦 {repo.name}</span>
                        <span className="repo-meta">
                          {repo.indexed ? '✅ indexed' : '⏳ not indexed'}
                        </span>
                      </div>
                    ))}
                    {repos.length === 0 && (
                      <div className="repo-empty">No repositories yet. Add a server path above.</div>
                    )}
                  </div>
                </div>
              </div>

              {/* Keyboard Shortcuts Documentation */}
              <div className="settings-section">
                <div className="setting-item vertical">
                  <span className="setting-label">Keyboard Shortcuts</span>
                  <ul className="shortcuts-list">
                    <li>
                      <code>Ctrl + Space</code> / <code>Cmd + Space</code> — Toggle microphone recording
                    </li>
                    <li>
                      <code>Enter</code> — Send query
                    </li>
                    <li>
                      <code>Shift + Enter</code> — New line in text field
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="settings-footer">
              <button
                type="button"
                className="settings-done-btn"
                onClick={() => setIsOpen(false)}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}