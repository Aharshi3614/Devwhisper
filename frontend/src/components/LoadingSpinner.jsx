import './LoadingSpinner.css'

export default function LoadingSpinner({ 
  size = 'medium', 
  label = '', 
  className = '' 
}) {
  const validSize = ['small', 'medium', 'large'].includes(size) ? size : 'medium'

  return (
    <div 
      className={`loading-spinner-container size-${validSize} ${className}`}
      role="status"
      aria-live="polite"
    >
      <div className="spinner-ring" aria-hidden="true" />
      {label && <span className="spinner-label">{label}</span>}
    </div>
  )
}
