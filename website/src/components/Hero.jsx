import './Hero.css'

export default function Hero() {
  return (
    <section className="hero">
      <div className="container hero-content">
        <div className="hero-trust">
          <span className="trust-dot"></span>
          <span className="mono">AI-ASSISTED DIAGNOSTIC INSTRUMENT</span>
        </div>

        <h1 className="hero-title">
          Brain MRI analysis<br />
          <span className="hero-subtitle-line">with visual explanation</span>
        </h1>

        <p className="hero-desc">
          Upload a brain MRI scan to receive an AI-powered classification with
          confidence scoring, uncertainty assessment, and a visual heatmap showing
          which regions influenced the analysis.
        </p>

        <div className="hero-actions">
          <a href="#analyze" className="btn-primary">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="1.6"/>
              <path d="M12 8v8M8 12h8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
            </svg>
            Upload &amp; analyze
          </a>
          <a href="#how-it-works" className="btn-secondary">How it works</a>
        </div>

        <div className="hero-notes">
          <div className="hero-note">
            <span className="note-tick" aria-hidden="true"></span>
            <span>Supports DICOM &amp; standard image formats</span>
          </div>
          <div className="hero-note">
            <span className="note-tick" aria-hidden="true"></span>
            <span>Results include uncertainty scoring</span>
          </div>
          <div className="hero-note">
            <span className="note-tick" aria-hidden="true"></span>
            <span>Visual heatmap for clinical transparency</span>
          </div>
        </div>
      </div>
    </section>
  )
}
