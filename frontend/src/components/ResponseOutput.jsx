import { useState } from 'react'

function ResponseOutput({ response, loading, error, onRetry }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    if (!response) return
    navigator.clipboard.writeText(response)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // A custom lightweight formatter for basic markdown (inline code and code blocks)
  const formatResponse = (text) => {
    if (!text) return null

    // Determine if we have an unclosed code block at the end (odd number of ``` occurrences)
    const occurrences = (text.match(/```/g) || []).length
    const isUnclosed = occurrences % 2 !== 0

    let parsedText = text
    if (isUnclosed) {
      // Append temporary closing triple-backtick to render the block correctly while streaming
      parsedText += '\n```'
    }

    const parts = parsedText.split(/(```[\s\S]*?```)/g)
    return parts.map((part, index) => {
      if (part.startsWith('```')) {
        const match = part.match(/```(\w*)\n([\s\S]*?)```/)
        const language = match ? match[1] : ''
        const code = match ? match[2] : part.slice(3, -3)
        return (
          <pre key={index} className="code-block">
            {language && <span className="code-lang">{language}</span>}
            <code>{code.trim()}</code>
          </pre>
        )
      }

      const inlineParts = part.split(/(`[^`\n]+`)/g)
      return (
        <span key={index}>
          {inlineParts.map((subPart, subIndex) => {
            if (subPart.startsWith('`') && subPart.endsWith('`')) {
              return (
                <code key={subIndex} className="inline-code">
                  {subPart.slice(1, -1)}
                </code>
              )
            }
            return subPart
          })}
        </span>
      )
    })
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
        {formatResponse(response)}
        {loading && <span className="streaming-cursor"></span>}
      </div>
    </div>
  )
}

export default ResponseOutput
