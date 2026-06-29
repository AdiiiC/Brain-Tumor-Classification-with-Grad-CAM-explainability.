import './Hero.css'

export default function Hero() {
  return (
    <section className="hero">
      <div className="container hero-content">
        <div className="hero-trust">
          <span className="trust-dot"></span>
          AI-Assisted Diagnostic Tool
        </div>

        <h1 className="hero-title">
          Brain MRI Analysis<br />
          <span className="hero-subtitle-line">with Visual Explanation</span>
        </h1>

        <p className="hero-desc">
          Upload a brain MRI scan to receive an AI-powered classification with
          confidence scoring, uncertainty assessment, and a visual heatmap showing
          which regions influenced the analysis.
        </p>

        <div className="hero-actions">
          <a href="#analyze" className="btn-primary">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
            </svg>
            Upload & Analyze
          </a>
          <a href="#how-it-works" className="btn-secondary">Learn How It Works</a>
        </div>

        <div className="hero-notes">
          <div className="hero-note">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--success)" strokeWidth="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg>
            <span>Supports DICOM & standard image formats</span>
          </div>
          <div className="hero-note">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--success)" strokeWidth="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg>
            <span>Results include uncertainty scoring</span>
          </div>
          <div className="hero-note">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--success)" strokeWidth="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg>
            <span>Visual heatmap for clinical transparency</span>
          </div>
        </div>
      </div>
    </section>
  )
}
