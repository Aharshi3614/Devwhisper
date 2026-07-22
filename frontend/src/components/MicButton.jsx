import { useState } from 'react'
import './MicButton.css'

export default function MicButton() {
  const [isRecording, setIsRecording] = useState(false)

  const handleClick = () => {
    setIsRecording(!isRecording)
  }

  return (
    <button
      className={`mic-button ${isRecording ? 'recording' : ''}`}
      onClick={handleClick}
      aria-label={isRecording ? 'Stop recording' : 'Start recording'}
    >
      <span className="mic-icon">🎙️</span>
      <span className="mic-text">
        {isRecording ? 'Stop Recording' : 'Start Recording'}
      </span>
    </button>
  )
}