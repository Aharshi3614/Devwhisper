import { useEffect, useState } from 'react'

import './BackendStatus.css'

const POLL_INTERVAL_MS = 10_000
const REQUEST_TIMEOUT_MS = 4_000

function BackendStatus() {
  const [status, setStatus] = useState('checking')
  const [diagnostics, setDiagnostics] = useState(null)
  const [showModal, setShowModal] = useState(false)

  useEffect(() => {
    let active = true
    let intervalId

    const checkBackend = async () => {
      const controller = new AbortController()
      const timeoutId = window.setTimeout(
        () => controller.abort(),
        REQUEST_TIMEOUT_MS,
      )

      try {
        const response = await fetch('/health/diagnostics', {
          method: 'GET',
          cache: 'no-store',
          signal: controller.signal,
        })

        if (!response.ok) {
          throw new Error(`Health check returned HTTP ${response.status}`)
        }

        const data = await response.json()
        if (active) {
          setDiagnostics(data)
          setStatus(data.status === 'healthy' ? 'online' : 'degraded')
        }
      } catch {
        if (active) {
          setStatus('offline')
          setDiagnostics(null)
        }
      } finally {
        window.clearTimeout(timeoutId)
      }
    }

    checkBackend()
    intervalId = window.setInterval(checkBackend, POLL_INTERVAL_MS)

    return () => {
      active = false
      window.clearInterval(intervalId)
    }
  }, [])

  const label = {
    checking: 'Checking backend',
    online: 'Backend online',
    degraded: 'Backend degraded',
    offline: 'Backend offline',
  }[status]

  return (
    <>
      <div
        className={`backend-status backend-status--${status}`}
        role="button"
        tabIndex={0}
        onClick={() => setShowModal(true)}
        aria-live="polite"
        style={{ cursor: 'pointer' }}
        title="Click to view subsystem health diagnostics"
      >
        <span className="backend-status__dot" aria-hidden="true" />
        <span>{label}</span>
      </div>

      {showModal && (
        <div
          className="diagnostics-modal-backdrop"
          onClick={() => setShowModal(false)}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
        >
          <div
            className="diagnostics-modal"
            onClick={(e) => e.stopPropagation()}
            style={{
              background: 'var(--bg-card, #1e1e2e)',
              color: 'var(--text, #fff)',
              padding: '1.5rem',
              borderRadius: '8px',
              maxWidth: '420px',
              width: '90%',
              boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0 }}>Subsystem Health</h3>
              <button onClick={() => setShowModal(false)} style={{ background: 'none', border: 'none', color: '#aaa', cursor: 'pointer', fontSize: '1.1rem' }}>✕</button>
            </div>
            {diagnostics ? (
              <div style={{ fontSize: '0.9rem', lineHeight: '1.6' }}>
                <div><strong>Status:</strong> <span className={`status-${status}`}>{status.toUpperCase()}</span></div>
                <hr style={{ borderColor: 'rgba(255,255,255,0.1)', margin: '0.8rem 0' }} />
                {diagnostics.subsystems && Object.entries(diagnostics.subsystems).map(([name, sub]) => (
                  <div key={name} style={{ marginBottom: '8px' }}>
                    <div>{sub.healthy ? '✅' : '⚠️'} <strong>{name.toUpperCase()}:</strong> {sub.healthy ? 'Operational' : 'Unavailable'}</div>
                    <div style={{ fontSize: '0.8rem', color: '#aaa', marginLeft: '1.4rem' }}>{sub.message}</div>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: '#ff6b6b' }}>Backend server is unreachable.</p>
            )}
          </div>
        </div>
      )}
    </>
  )
}

export default BackendStatus
