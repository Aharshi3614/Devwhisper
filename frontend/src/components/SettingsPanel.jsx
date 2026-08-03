import { useState, useEffect, useCallback } from 'react'
import ThemeToggle from './ThemeToggle.jsx'
import './SettingsPanel.css'

export default function SettingsPanel() {
  const [isOpen, setIsOpen] = useState(false)
  
  // Future setting state placeholders (stored in localStorage for extension)
  const [autoSubmitVoice, setAutoSubmitVoice] = useState(() => {
    return localStorage.getItem('devwhisper_auto_submit') !== 'false'
  })

  const [progress, setProgress] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [uploadSuccess, setUploadSuccess] = useState('')
  const [dragActive, setDragActive] = useState(false)

  useEffect(() => {
    localStorage.setItem('devwhisper_auto_submit', autoSubmitVoice)
  }, [autoSubmitVoice])

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

      setUploadSuccess(data.message || 'ZIP uploaded and extraction started.')
      setUploadError('')
    } catch (err) {
      setUploadError(err.message || 'An error occurred during upload.')
      setUploadSuccess('')
    } finally {
      setUploading(false)
    }
  }, [])

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
    } catch (err) {
      setUploadError(err.message || 'Failed to start indexing.')
    }
  }, [progress])

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
                    </div>
                  )}
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