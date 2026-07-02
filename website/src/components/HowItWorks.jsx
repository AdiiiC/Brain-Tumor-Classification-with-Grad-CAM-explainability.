import './HowItWorks.css'

const steps = [
  {
    num: '1',
    title: 'Upload scan',
    desc: 'Upload any brain MRI image — the system accepts standard formats (JPG, PNG) as well as clinical DICOM files directly from imaging equipment.',
    icon: (
      <svg width="30" height="30" viewBox="0 0 32 32" fill="none" aria-hidden="true">
        <rect x="6" y="6" width="20" height="20" rx="2" stroke="var(--primary)" strokeWidth="1.6"/>
        <path d="M16 11v8M12 15h8" stroke="var(--primary)" strokeWidth="1.6" strokeLinecap="round"/>
      </svg>
    ),
  },
  {
    num: '2',
    title: 'AI classification',
    desc: 'The scan is preprocessed (enhanced, cropped, normalized) and analyzed by a deep learning model trained on thousands of clinical MRI images.',
    icon: (
      <svg width="30" height="30" viewBox="0 0 32 32" fill="none" aria-hidden="true">
        <circle cx="16" cy="16" r="4" stroke="var(--primary)" strokeWidth="1.6"/>
        <path d="M16 4v6M16 22v6M4 16h6M22 16h6M8 8l4 4M20 20l4 4M24 8l-4 4M12 20l-4 4" stroke="var(--primary)" strokeWidth="1.4" strokeLinecap="round"/>
      </svg>
    ),
  },
  {
    num: '3',
    title: 'Uncertainty check',
    desc: 'The system runs 30 independent analyses with slight variations. If results are inconsistent, it flags the case as uncertain — recommending specialist review.',
    icon: (
      <svg width="30" height="30" viewBox="0 0 32 32" fill="none" aria-hidden="true">
        <path d="M6 22c2-8 4-12 5-12s1.5 6 3 6 2-10 3-10 2 8 4 16" stroke="var(--primary)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M4 26h24" stroke="var(--text-muted)" strokeWidth="1.4" strokeLinecap="round"/>
      </svg>
    ),
  },
  {
    num: '4',
    title: 'Visual explanation',
    desc: 'A heatmap overlay shows exactly which brain regions the AI examined most closely. This allows you to verify the analysis aligns with clinical findings.',
    icon: (
      <svg width="30" height="30" viewBox="0 0 32 32" fill="none" aria-hidden="true">
        <rect x="6" y="6" width="20" height="20" rx="2" stroke="var(--primary)" strokeWidth="1.6"/>
        <circle cx="16" cy="16" r="5" stroke="var(--accent)" strokeWidth="1.6"/>
        <circle cx="16" cy="16" r="1.6" fill="var(--accent)"/>
      </svg>
    ),
  },
]

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="how-section">
      <div className="container">
        <div className="section-header">
          <span className="eyebrow">02 / Pipeline</span>
          <h2 className="section-title">How it works</h2>
          <p className="section-subtitle">
            A four-step process from scan upload to visual explanation — designed for clinical clarity.
          </p>
        </div>

        <div className="steps-grid">
          {steps.map((step) => (
            <div className="step-card card reticle" key={step.num}>
              <div className="step-header">
                <div className="step-icon">{step.icon}</div>
                <span className="step-num mono">Step {step.num}</span>
              </div>
              <h3 className="step-title">{step.title}</h3>
              <p className="step-desc">{step.desc}</p>
            </div>
          ))}
        </div>

        <div className="disclaimer-box">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <rect x="3" y="3" width="18" height="18" rx="2" stroke="var(--warning)" strokeWidth="1.6"/>
            <path d="M12 8v5M12 16h.01" stroke="var(--warning)" strokeWidth="1.8" strokeLinecap="round"/>
          </svg>
          <p>
            <strong>Important:</strong> This tool is designed to assist clinical decision-making, not replace it.
            All results should be reviewed by a qualified medical professional before any clinical action is taken.
          </p>
        </div>
      </div>
    </section>
  )
}
