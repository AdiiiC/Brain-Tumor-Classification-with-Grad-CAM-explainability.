import './HowItWorks.css'

const steps = [
  {
    num: '1',
    title: 'Upload Scan',
    desc: 'Upload any brain MRI image — the system accepts standard formats (JPG, PNG) as well as clinical DICOM files directly from imaging equipment.',
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
      </svg>
    ),
  },
  {
    num: '2',
    title: 'AI Classification',
    desc: 'The scan is preprocessed (enhanced, cropped, normalized) and analyzed by a deep learning model trained on thousands of clinical MRI images.',
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2a7 7 0 017 7c0 2.5-1.5 5-3 6.5V18H8v-2.5C6.5 14 5 11.5 5 9a7 7 0 017-7z"/>
        <path d="M9 22h6"/>
      </svg>
    ),
  },
  {
    num: '3',
    title: 'Uncertainty Check',
    desc: 'The system runs 30 independent analyses with slight variations. If results are inconsistent, it flags the case as uncertain — recommending specialist review.',
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/>
        <path d="M12 16v-4M12 8h.01"/>
      </svg>
    ),
  },
  {
    num: '4',
    title: 'Visual Explanation',
    desc: 'A heatmap overlay shows exactly which brain regions the AI examined most closely. This allows you to verify the analysis aligns with clinical findings.',
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
        <circle cx="12" cy="12" r="4"/>
        <path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>
      </svg>
    ),
  },
]

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="how-section">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">How It Works</h2>
          <p className="section-subtitle">
            A four-step process from scan upload to visual explanation — designed for clinical clarity.
          </p>
        </div>

        <div className="steps-grid">
          {steps.map((step) => (
            <div className="step-card card" key={step.num}>
              <div className="step-header">
                <div className="step-icon">{step.icon}</div>
                <span className="step-num">Step {step.num}</span>
              </div>
              <h3 className="step-title">{step.title}</h3>
              <p className="step-desc">{step.desc}</p>
            </div>
          ))}
        </div>

        <div className="disclaimer-box">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--warning)" strokeWidth="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
            <path d="M12 9v4M12 17h.01"/>
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
