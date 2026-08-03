import { useEffect, useState } from 'react'

import './BackendStatus.css'

const POLL_INTERVAL_MS = 10_000
const REQUEST_TIMEOUT_MS = 4_000

function BackendStatus() {
  const [status, setStatus] = useState('checking')

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
        const response = await fetch('/health', {
          method: 'GET',
          cache: 'no-store',
          signal: controller.signal,
        })

        if (!response.ok) {
          throw new Error(`Health check returned HTTP ${response.status}`)
        }

        if (active) {
          setStatus('online')
        }
      } catch {
        if (active) {
          setStatus('offline')
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
    offline: 'Backend offline',
  }[status]

  return (
    <div
      className={`backend-status backend-status--${status}`}
      role="status"
      aria-live="polite"
      title={
        status === 'offline'
          ? 'The DevWhisper backend could not be reached. Retrying automatically.'
          : label
      }
    >
      <span className="backend-status__dot" aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}

export default BackendStatus
