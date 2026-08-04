import { useEffect } from 'react'
import './MicButton.css'

export default function MicButton({ isListening, onClick, disabled, countdown }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Check if Ctrl + Space or Cmd + Space is pressed
      if ((e.ctrlKey || e.metaKey) && e.code === 'Space') {
        const activeElement = document.activeElement
        const isInputElement =
          activeElement &&
          (activeElement.tagName === 'INPUT' ||
            activeElement.tagName === 'TEXTAREA' ||
            activeElement.isContentEditable)

        // Prevent triggering while actively typing in an input
        if (isInputElement) return

        e.preventDefault()
        if (!disabled && onClick) {
          onClick()
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClick, disabled])

  const labelText = isListening ? 'Stop voice recording' : 'Start voice recording'

  return (
    <div className="mic-button-wrapper">
      <button
        type="button"
        className={`mic-button ${isListening ? 'listening' : ''}`}
        onClick={onClick}
        disabled={disabled}
        aria-label={labelText}
        title={`${labelText} (Ctrl + Space)`}
      >
        <span className="mic-icon">🎙️</span>
        {isListening && (
          <>
            <span className="pulse-ring-glow ring-1"></span>
            <span className="pulse-ring-glow ring-2"></span>
            <span className="pulse-ring-glow ring-3"></span>
          </>
        )}
        {isListening && countdown !== null && (
          <span className="mic-countdown" aria-live="polite">{countdown}s</span>
        )}
      </button>
      <span className="mic-shortcut-hint">Ctrl + Space</span>
    </div>
  )
}
