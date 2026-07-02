import './Trust.css'

const stats = [
  { value: '96.4%', label: 'Classification Accuracy', desc: 'On held-out test data' },
  { value: '21,732', label: 'Training Images', desc: 'Balanced multi-source MRI dataset' },
  { value: '4', label: 'Tumor Categories', desc: 'Glioma, Meningioma, Pituitary, None' },
  { value: '50×', label: 'MC Dropout Passes', desc: 'For uncertainty estimation' },
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
          <span className="eyebrow">05 / Assurance</span>
          <h2 className="section-title">Safety &amp; accuracy</h2>
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
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M12 3l7 3v5c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6z" stroke="var(--primary)" strokeWidth="1.6" strokeLinejoin="round"/>
                <path d="M9 12l2 2 4-4" stroke="var(--primary)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <div>
                <h4>{sg.title}</h4>
                <p>{sg.desc}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Limitations — Addressed */}
        <div className="limitations-box resolved">
          <h3>Previously Known Limitations — Now Addressed</h3>
          <ul>
            <li>
              <strong>Multi-sequence support:</strong> Model now trained on all 4 MRI
              sequences (T1, T1-CE, T2, FLAIR) from BraTS 2021 — 2,000 images per
              sequence integrated into training data. Auto-detection adapts preprocessing
              per sequence type.
              <span className="fixed-badge">Trained / 4 sequences</span>
            </li>
            <li>
              <strong>Small tumor detection:</strong> Dedicated patch classifier trained
              on multi-scale patches (60px, 90px, 120px) from tumor vs. clean brain
              regions. Uses sliding window with NMS to detect lesions as small as ~3mm.
              <span className="fixed-badge">Trained / patch_classifier.keras</span>
            </li>
            <li>
              <strong>WHO tumor grading:</strong> Binary grade classifier (HGG vs LGG)
              trained on DICOM-multi dataset (105 patients with explicit WHO grade labels)
              + LGG segmentation (111 patients) + BraTS high-grade data. Combined with
              image feature analysis for Grade I–IV estimation.
              <span className="fixed-badge">Trained / grade_classifier.keras</span>
            </li>
            <li>
              <strong>Pediatric support:</strong> Bayesian re-weighting using published
              pediatric neuro-oncology epidemiology (CBTRUS/WHO). Actually adjusts model
              probabilities: P(class|image, age) ∝ P(class|image) × P(class|age) / P(class|adult).
              Includes age-group-specific priors, differentials, and workup recommendations.
              <span className="fixed-badge">Fixed / Bayesian priors</span>
            </li>
            <li>
              <strong>Image quality gating:</strong> Pre-inference quality assessment scores
              resolution, blur (Laplacian), SNR, compression artifacts, and brain coverage —
              low-quality scans are flagged with specific improvement recommendations.
              <span className="fixed-badge">Fixed / assess/quality</span>
            </li>
          </ul>
        </div>
      </div>
    </section>
  )
}
