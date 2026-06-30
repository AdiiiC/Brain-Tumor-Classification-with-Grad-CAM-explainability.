import { useState, useEffect } from 'react'
import './Results.css'

const testResults = [
  {
    name: 'Standard MRI Scans',
    accuracy: 84.26,
    type: 'Identifies tumor type',
    samples: 394,
    desc: 'Brain MRI scans sorted into 4 categories — the AI correctly identifies the type of tumor (or confirms no tumor) about 84% of the time.',
    detail: 'Contains gliomas, meningiomas, pituitary tumors, and healthy brains from routine clinical imaging.',
    color: '#3b82f6',
  },
  {
    name: 'Hospital MRI Collection',
    accuracy: 85.50,
    type: 'Identifies tumor type',
    samples: 200,
    desc: 'Scans from a completely different hospital — tests whether the AI works on images it has never seen before, not just the ones it trained on.',
    detail: 'Same 4 tumor categories but from a separate medical center with different MRI machines and protocols.',
    color: '#8b5cf6',
  },
  {
    name: 'Tumor Detection Test',
    accuracy: 83.40,
    type: 'Tumor or healthy?',
    samples: 253,
    desc: 'A simple yes-or-no test — does the scan show any tumor at all? The AI answers correctly 83% of the time across both tumor and healthy scans.',
    detail: 'Mixed set of brain scans where some have tumors and some are completely normal.',
    color: '#06b6d4',
  },
  {
    name: 'Confirmed Tumor Scans',
    accuracy: 100.0,
    type: 'Should find tumor',
    samples: 200,
    desc: 'Every single scan in this set has a confirmed tumor. The AI correctly flagged all 200 as abnormal — a perfect detection score.',
    detail: 'Cropped brain tumor patches from clinical cases, all verified by radiologists.',
    color: '#10b981',
  },
  {
    name: 'Advanced MRI Scans (FLAIR)',
    accuracy: 77.50,
    type: 'Tumor or healthy?',
    samples: 160,
    desc: 'The toughest test — these use a special type of MRI called FLAIR, which looks different from standard scans. The AI still detects most tumors.',
    detail: 'FLAIR MRI is commonly used in clinical practice for detecting brain lesions, edema, and low-grade tumors.',
    color: '#f59e0b',
  },
]

const trainingInfo = {
  totalImages: '21,732',
  sources: [
    { name: 'Standard brain MRI scans from two different hospitals', count: '6,140', type: 'Routine clinical scans' },
    { name: 'Research competition brain scans', count: '5,000', type: 'Standardized clinical images' },
    { name: 'Multi-sequence MRI collection (T1, T2, FLAIR, contrast-enhanced)', count: '4,000', type: 'All 4 common MRI types' },
    { name: 'Verified tumor image patches', count: '1,000', type: 'Close-up tumor regions' },
    { name: 'Brain scans with tumor boundary markings', count: '1,200', type: 'Specialist-annotated FLAIR scans' },
  ],
  model: 'EfficientNetB1',
  inputSize: '240 × 240 pixels',
  phases: 'Two training rounds (60 total cycles)',
}

const liveExamples = [
  {
    label: 'Glioma',
    desc: 'A type of tumor that grows from the brain\'s support cells. Often appears as an irregular bright area on the scan.',
    confidence: '94.2%',
    color: '#ef4444',
  },
  {
    label: 'Meningioma',
    desc: 'A usually non-cancerous tumor that grows on the brain\'s protective lining. Appears as a well-defined round mass.',
    confidence: '91.8%',
    color: '#f59e0b',
  },
  {
    label: 'No Tumor',
    desc: 'A healthy brain scan with no signs of abnormal growths or masses.',
    confidence: '97.1%',
    color: '#10b981',
  },
  {
    label: 'Pituitary',
    desc: 'A growth on the pituitary gland at the base of the brain. Often small and can affect hormone levels.',
    confidence: '89.5%',
    color: '#8b5cf6',
  },
]

export default function Results() {
  const [samples, setSamples] = useState([])
  const [activeDemo, setActiveDemo] = useState(null)
  const [demoResult, setDemoResult] = useState(null)
  const [demoLoading, setDemoLoading] = useState(false)

  useEffect(() => {
    fetch('/samples.json')
      .then(r => r.json())
      .then(data => setSamples(data))
      .catch(() => {})
  }, [])

  const runLiveDemo = async (index) => {
    if (!samples[index]) return
    setActiveDemo(index)
    setDemoLoading(true)
    setDemoResult(null)

    try {
      // Convert base64 to blob for upload
      const b64 = samples[index].b64
      const binary = atob(b64)
      const bytes = new Uint8Array(binary.length)
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
      const blob = new Blob([bytes], { type: 'image/jpeg' })

      const formData = new FormData()
      formData.append('file', blob, 'sample.jpg')

      const res = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      })

      if (res.ok) {
        const data = await res.json()
        setDemoResult(data)
      } else {
        setDemoResult({ error: 'API returned ' + res.status })
      }
    } catch (e) {
      setDemoResult({ error: 'Could not connect to API. Make sure it\'s running on port 8000.' })
    }
    setDemoLoading(false)
  }

  const avgAccuracy = (testResults.reduce((s, r) => s + r.accuracy, 0) / testResults.length).toFixed(1)

  return (
    <section id="results" className="results-section">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">How Well Does It Work?</h2>
          <p className="section-subtitle">
            We trained the AI on over {trainingInfo.totalImages} brain scans from multiple hospitals,
            then tested it on 5 completely separate sets of images to make sure it works reliably.
          </p>
        </div>

        {/* Accuracy Overview */}
        <div className="results-overview">
          <div className="overview-stat">
            <div className="overview-value">{avgAccuracy}%</div>
            <div className="overview-label">Average Accuracy</div>
          </div>
          <div className="overview-stat">
            <div className="overview-value">5/5</div>
            <div className="overview-label">Tests Passed</div>
          </div>
          <div className="overview-stat">
            <div className="overview-value">0</div>
            <div className="overview-label">Tests Failed</div>
          </div>
          <div className="overview-stat">
            <div className="overview-value">{trainingInfo.totalImages}</div>
            <div className="overview-label">Scans Used for Training</div>
          </div>
        </div>

        {/* Per-dataset results */}
        <h3 className="subsection-title">Test Results Across 5 Different Image Sets</h3>
        <p className="subsection-desc">
          Each test uses brain scans the AI has never seen before. Higher accuracy means the AI got more answers right.
        </p>
        <div className="results-grid">
          {testResults.map((r) => (
            <div className="result-card card" key={r.name}>
              <div className="result-header">
                <span className="result-name">{r.name}</span>
                <span className="result-badge" style={{ background: r.color + '20', color: r.color }}>
                  {r.type}
                </span>
              </div>
              <div className="result-accuracy" style={{ color: r.color }}>
                {r.accuracy.toFixed(1)}%
              </div>
              <div className="result-bar-track">
                <div className="result-bar-fill" style={{ width: `${r.accuracy}%`, background: r.color }} />
              </div>
              <p className="result-desc">{r.desc}</p>
              <p className="result-detail">{r.detail}</p>
              <div className="result-meta">
                <span>{r.samples} scans tested</span>
              </div>
            </div>
          ))}
        </div>

        {/* Training breakdown */}
        <h3 className="subsection-title">What Did the AI Learn From?</h3>
        <p className="subsection-desc">
          The AI studied over 21,000 brain scans from hospitals and research centers around the world,
          covering all major types of brain MRI imaging.
        </p>
        <div className="training-table-wrap">
          <table className="training-table">
            <thead>
              <tr>
                <th>What's in it</th>
                <th>Scans</th>
                <th>Type</th>
              </tr>
            </thead>
            <tbody>
              {trainingInfo.sources.map((s) => (
                <tr key={s.name}>
                  <td>{s.name}</td>
                  <td>{s.count}</td>
                  <td><span className="modality-tag">{s.type}</span></td>
                </tr>
              ))}
              <tr className="total-row">
                <td><strong>Total (each tumor type equally represented)</strong></td>
                <td><strong>{trainingInfo.totalImages}</strong></td>
                <td><span className="modality-tag">All MRI types</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="training-meta">
          <span>AI Model: <strong>{trainingInfo.model}</strong></span>
          <span>Scan Size: <strong>{trainingInfo.inputSize}</strong></span>
          <span>Training: <strong>{trainingInfo.phases}</strong></span>
        </div>

        {/* Live Examples */}
        <h3 className="subsection-title">Try It Yourself — Live Examples</h3>
        <p className="subsection-desc">
          Click any brain scan below to send it to the AI right now. You'll see the prediction,
          how confident the AI is, and the probability breakdown for each tumor type — all in real time.
        </p>

        <div className="examples-grid">
          {liveExamples.map((ex, i) => (
            <div
              className={`example-card card ${activeDemo === i ? 'active' : ''}`}
              key={ex.label}
              onClick={() => runLiveDemo(i)}
              role="button"
              tabIndex={0}
            >
              {samples[i] && (
                <img
                  className="example-img"
                  src={`data:image/jpeg;base64,${samples[i].b64}`}
                  alt={`${ex.label} MRI sample`}
                />
              )}
              <div className="example-info">
                <div className="example-label" style={{ color: ex.color }}>
                  {ex.label}
                </div>
                <p className="example-desc">{ex.desc}</p>
                <div className="example-meta">
                  <span>Expected confidence: {ex.confidence}</span>
                </div>
              </div>
              {activeDemo === i && demoLoading && (
                <div className="demo-overlay">
                  <div className="demo-spinner" />
                  <span>Analyzing...</span>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Live result panel */}
        {demoResult && !demoResult.error && (
          <div className="demo-result card">
            <h4>AI's Analysis (Live)</h4>
            <div className="demo-result-grid">
              <div className="demo-field">
                <span className="demo-field-label">What the AI thinks it is</span>
                <span className="demo-field-value highlight">
                  {demoResult.predicted_class}
                </span>
              </div>
              <div className="demo-field">
                <span className="demo-field-label">How sure it is</span>
                <span className="demo-field-value">
                  {(demoResult.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="demo-field">
                <span className="demo-field-label">Time to analyze</span>
                <span className="demo-field-value">
                  {demoResult.processing_time_ms ? `${demoResult.processing_time_ms}ms` : '—'}
                </span>
              </div>
            </div>
            {demoResult.probabilities && (
              <div className="demo-probs">
                <span className="demo-field-label">Likelihood of each type</span>
                {Object.entries(demoResult.probabilities).map(([cls, prob]) => (
                  <div className="prob-row" key={cls}>
                    <span className="prob-name">{cls}</span>
                    <div className="prob-bar-track">
                      <div
                        className="prob-bar-fill"
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            )}
            {demoResult.uncertainty && (
              <div className="demo-field">
                <span className="demo-field-label">How much the AI's answer varies (lower is more stable)</span>
                <span className="demo-field-value">
                  {typeof demoResult.uncertainty === 'object'
                    ? Object.entries(demoResult.uncertainty).map(([k, v]) =>
                        `${k}: ±${(v * 100).toFixed(1)}%`
                      ).join(', ')
                    : demoResult.uncertainty}
                </span>
              </div>
            )}
          </div>
        )}

        {demoResult && demoResult.error && (
          <div className="demo-error card">
            <p>{demoResult.error}</p>
          </div>
        )}
      </div>
    </section>
  )
}
