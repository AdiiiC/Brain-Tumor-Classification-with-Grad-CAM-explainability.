import './Interpret.css'

export default function Interpret() {
  return (
    <section id="interpret">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">Reading Your Results</h2>
          <p className="section-subtitle">
            A guide to understanding the analysis output — what each section means clinically.
          </p>
        </div>

        <div className="interpret-grid">
          <div className="card interpret-card">
            <div className="interpret-icon green">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg>
            </div>
            <h3>Classification</h3>
            <p>
              The system identifies one of four categories: <strong>Glioma</strong>,
              <strong> Meningioma</strong>, <strong>Pituitary Tumor</strong>, or
              <strong> No Tumor</strong>. The classification represents the most likely finding
              based on MRI patterns.
            </p>
          </div>

          <div className="card interpret-card">
            <div className="interpret-icon blue">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 20V10M18 20V4M6 20v-4"/></svg>
            </div>
            <h3>Confidence Score</h3>
            <p>
              A percentage indicating how certain the AI is. <strong>Above 90%</strong> = high confidence.
              <strong> 70-90%</strong> = moderate — consider correlating with clinical findings.
              <strong> Below 70%</strong> = low confidence — specialist review recommended.
            </p>
          </div>

          <div className="card interpret-card">
            <div className="interpret-icon amber">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/><path d="M12 8v4M12 16h.01"/></svg>
            </div>
            <h3>Uncertainty (±%)</h3>
            <p>
              Measures how consistent the AI's assessment is across multiple internal analyses.
              <strong> Below ±3%</strong> = highly consistent.
              <strong> Above ±5%</strong> = variable — the system will flag this case for manual review.
            </p>
          </div>

          <div className="card interpret-card">
            <div className="interpret-icon purple">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="12" cy="12" r="4"/></svg>
            </div>
            <h3>Visual Heatmap</h3>
            <p>
              The colored overlay shows which brain regions most influenced the classification.
              <strong> Red/yellow areas</strong> = high importance.
              <strong> Blue/green areas</strong> = low importance.
              Verify these align with clinically suspicious regions.
            </p>
          </div>

          <div className="card interpret-card">
            <div className="interpret-icon red">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><path d="M12 9v4M12 17h.01"/></svg>
            </div>
            <h3>Review Flags</h3>
            <p>
              Cases are automatically flagged when confidence is low or uncertainty is high.
              A flagged result means: <strong>do not rely on this assessment alone</strong> — 
              additional imaging or specialist consultation is recommended.
            </p>
          </div>

          <div className="card interpret-card">
            <div className="interpret-icon teal">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>
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
