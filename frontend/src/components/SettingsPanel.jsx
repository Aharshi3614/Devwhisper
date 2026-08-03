import { useState, useEffect } from 'react'
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