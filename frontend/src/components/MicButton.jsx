import './MicButton.css'

export default function MicButton({ isListening, onClick, disabled }) {
  return (
    <button
      type="button"
      className={`mic-button ${isListening ? 'listening' : ''}`}
      onClick={onClick}
      disabled={disabled}
      aria-label={isListening ? 'Stop voice recording' : 'Start voice recording'}
      title={isListening ? 'Stop voice recording' : 'Start voice recording'}
    >
      <span className="mic-icon">🎙️</span>
      {isListening && (
        <span className="pulse-ring-glow"></span>
      )}
    </button>
  )
}