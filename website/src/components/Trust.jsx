import './Trust.css'

const stats = [
  { value: '96.4%', label: 'Classification Accuracy', desc: 'On held-out test data' },
  { value: '3,264', label: 'Training Images', desc: 'Multi-institutional MRI dataset' },
  { value: '4', label: 'Tumor Categories', desc: 'Glioma, Meningioma, Pituitary, None' },
  { value: '30×', label: 'MC Dropout Passes', desc: 'For uncertainty estimation' },
]

const safeguards = [
  {
    title: 'Uncertainty Flagging',
    desc: 'When the model is not confident, cases are automatically flagged for specialist review. No ambiguous result is presented as definitive.',
  },
  {
    title: 'Explainability Built-In',
    desc: 'Every prediction includes a visual explanation (Grad-CAM++) showing which brain regions influenced the decision — allowing clinicians to verify.',
  },
  {
    title: 'Calibrated Confidence',
    desc: 'Confidence scores are temperature-scaled so that "90% confidence" truly means the model is correct 90% of the time — not artificially inflated.',
  },
  {
    title: 'Multi-Pass Verification',
    desc: 'Each scan is analyzed multiple times with slight internal variations (Monte Carlo Dropout + Test-Time Augmentation) to ensure consistency.',
  },
]

export default function Trust() {
  return (
    <section id="trust" className="trust-section">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">Safety & Accuracy</h2>
          <p className="section-subtitle">
            Built with clinical safety in mind — transparent about capabilities and limitations.
          </p>
        </div>

        {/* Stats */}
        <div className="trust-stats">
          {stats.map((s) => (
            <div className="stat-card card" key={s.label}>
              <div className="stat-value">{s.value}</div>
              <div className="stat-label">{s.label}</div>
              <div className="stat-desc">{s.desc}</div>
            </div>
          ))}
        </div>

        {/* Safeguards */}
        <div className="safeguards-grid">
          {safeguards.map((sg) => (
            <div className="safeguard-item" key={sg.title}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--success)" strokeWidth="2">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              </svg>
              <div>
                <h4>{sg.title}</h4>
                <p>{sg.desc}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Limitations */}
        <div className="limitations-box">
          <h3>Known Limitations</h3>
          <ul>
            <li>Trained primarily on T1-weighted contrast-enhanced MRI — performance on other sequences may vary.</li>
            <li>Cannot detect tumors smaller than approximately 5mm or in early stages with minimal contrast enhancement.</li>
            <li>Does not provide tumor grading (WHO Grade I-IV) — only classification by type.</li>
            <li>Not validated for pediatric brain tumors — trained on adult imaging data.</li>
            <li>Image quality matters — heavily compressed or low-resolution scans reduce accuracy.</li>
          </ul>
        </div>
      </div>
    </section>
  )
}
