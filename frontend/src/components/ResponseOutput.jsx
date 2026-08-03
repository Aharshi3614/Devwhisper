import { useState } from 'react'
import Markdown from './Markdown'

function ResponseOutput({ response, loading, error, onRetry }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    if (!response) return
    navigator.clipboard.writeText(response)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Show loading spinner ONLY before we have received any response text
  if (loading && !response) {
    return (
      <div className="response-container loading">
        <div className="loading-spinner-wrapper">
          <div className="loading-spinner"></div>
          <span className="loading-text">DevWhisper is thinking...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="response-container error">
        <div className="response-header error-header">
          <span className="error-title">⚠️ Query Failed</span>
          {onRetry && (
            <button onClick={onRetry} className="retry-button" title="Retry query">
              🔄 Retry Request
            </button>
          )}
        </div>
        <div className="error-message">{error}</div>
      </div>
    )
  }

  if (!response) {
    return null
  }

  return (
    <div className="response-container success">
      <div className="response-header">
        <span className="response-title">🎙️ DevWhisper Response</span>
        <button onClick={handleCopy} className="copy-button" title="Copy response to clipboard">
          {copied ? '✅ Copied' : '📋 Copy'}
        </button>
      </div>
      <div className="response-body">
        <Markdown text={response} />
        {loading && <span className="streaming-cursor"></span>}
      </div>
    </div>
  )
}

export default ResponseOutput
