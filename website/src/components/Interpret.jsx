import './Interpret.css'

export default function Interpret() {
  return (
    <section id="interpret">
      <div className="container">
        <div className="section-header">
          <span className="eyebrow">03 / Readout</span>
          <h2 className="section-title">Reading your results</h2>
          <p className="section-subtitle">
            A guide to understanding the analysis output — what each section means clinically.
          </p>
        </div>

        <div className="interpret-grid">
          <div className="card interpret-card reticle">
            <div className="interpret-icon signal">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="1.6"/><path d="M8 12l3 3 5-6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
            </div>
            <h3>Classification</h3>
            <p>
              The system identifies one of four categories: <strong>Glioma</strong>,
              <strong> Meningioma</strong>, <strong>Pituitary Tumor</strong>, or
              <strong> No Tumor</strong>. The classification represents the most likely finding
              based on MRI patterns.
            </p>
          </div>

          <div className="card interpret-card reticle">
            <div className="interpret-icon signal">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 20V9M10 20V4M16 20v-7M4 20h16" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/></svg>
            </div>
            <h3>Confidence Score</h3>
            <p>
              A percentage indicating how certain the AI is. <strong>Above 90%</strong> = high confidence.
              <strong> 70-90%</strong> = moderate — consider correlating with clinical findings.
              <strong> Below 70%</strong> = low confidence — specialist review recommended.
            </p>
          </div>

          <div className="card interpret-card reticle">
            <div className="interpret-icon accent">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 18c2-7 3-10 4-10s1 5 2 5 1.5-8 2-8 1.5 6 2 8 1 5 2 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/><path d="M3 21h18" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/></svg>
            </div>
            <h3>Uncertainty (±%)</h3>
            <p>
              Measures how consistent the AI's assessment is across multiple internal analyses.
              <strong> Below ±3%</strong> = highly consistent.
              <strong> Above ±5%</strong> = variable — the system will flag this case for manual review.
            </p>
          </div>

          <div className="card interpret-card reticle">
            <div className="interpret-icon accent">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="1.6"/><circle cx="12" cy="12" r="4.5" stroke="currentColor" strokeWidth="1.6"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/></svg>
            </div>
            <h3>Visual Heatmap</h3>
            <p>
              The colored overlay shows which brain regions most influenced the classification.
              <strong> Warm areas</strong> = high importance.
              <strong> Cool areas</strong> = low importance.
              Verify these align with clinically suspicious regions.
            </p>
          </div>

          <div className="card interpret-card reticle">
            <div className="interpret-icon danger">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="1.6"/><path d="M12 8v5M12 16h.01" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/></svg>
            </div>
            <h3>Review Flags</h3>
            <p>
              Cases are automatically flagged when confidence is low or uncertainty is high.
              A flagged result means: <strong>do not rely on this assessment alone</strong> — 
              additional imaging or specialist consultation is recommended.
            </p>
          </div>

          <div className="card interpret-card reticle">
            <div className="interpret-icon signal">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M6 3h8l4 4v14H6z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round"/><path d="M14 3v4h4M9 13h6M9 17h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
            </div>
            <h3>Clinical Recommendation</h3>
            <p>
              A plain-language summary of what the result suggests. This is not a diagnosis — 
              it provides context to help prioritize cases and inform next steps
              in the clinical workflow.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
