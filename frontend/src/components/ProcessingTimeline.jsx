import React from 'react'
import './ProcessingTimeline.css'

export default function ProcessingTimeline({ stages = [] }) {
  if (!stages || stages.length === 0) return null

  return (
    <div className="processing-timeline" data-testid="processing-timeline">
      <div className="timeline-title">Processing Pipeline</div>
      <div className="timeline-stages">
        {stages.map((stage, idx) => {
          let statusClass = 'pending'
          if (stage.status === 'completed') statusClass = 'completed'
          else if (stage.status === 'in_progress') statusClass = 'in-progress'
          else if (stage.status === 'failed') statusClass = 'failed'

          return (
            <div key={idx} className={`timeline-stage ${statusClass}`}>
              <div className="stage-icon">
                {stage.status === 'completed' && '✓'}
                {stage.status === 'in_progress' && <span className="spinner"></span>}
                {stage.status === 'failed' && '✗'}
                {stage.status === 'pending' && '○'}
              </div>
              <div className="stage-info">
                <span className="stage-label">{stage.label}</span>
                {stage.detail && <span className="stage-detail">{stage.detail}</span>}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
