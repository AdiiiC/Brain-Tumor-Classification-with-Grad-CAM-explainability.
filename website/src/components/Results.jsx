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
    tone: 'signal',
  },
  {
    name: 'Hospital MRI Collection',
    accuracy: 85.50,
    type: 'Identifies tumor type',
    samples: 200,
    desc: 'Scans from a completely different hospital — tests whether the AI works on images it has never seen before, not just the ones it trained on.',
    detail: 'Same 4 tumor categories but from a separate medical center with different MRI machines and protocols.',
    tone: 'signal',
  },
  {
    name: 'Tumor Detection Test',
    accuracy: 83.40,
    type: 'Tumor or healthy?',
    samples: 253,
    desc: 'A simple yes-or-no test — does the scan show any tumor at all? The AI answers correctly 83% of the time across both tumor and healthy scans.',
    detail: 'Mixed set of brain scans where some have tumors and some are completely normal.',
    tone: 'signal',
  },
  {
    name: 'Confirmed Tumor Scans',
    accuracy: 100.0,
    type: 'Should find tumor',
    samples: 200,
    desc: 'Every single scan in this set has a confirmed tumor. The AI correctly flagged all 200 as abnormal — a perfect detection score.',
    detail: 'Cropped brain tumor patches from clinical cases, all verified by radiologists.',
    tone: 'strong',
  },
  {
    name: 'Advanced MRI Scans (FLAIR)',
    accuracy: 77.50,
    type: 'Tumor or healthy?',
    samples: 160,
    desc: 'The toughest test — these use a special type of MRI called FLAIR, which looks different from standard scans. The AI still detects most tumors.',
    detail: 'FLAIR MRI is commonly used in clinical practice for detecting brain lesions, edema, and low-grade tumors.',
    tone: 'accent',
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
  },
  {
    label: 'Meningioma',
    desc: 'A usually non-cancerous tumor that grows on the brain\'s protective lining. Appears as a well-defined round mass.',
    confidence: '91.8%',
  },
  {
    label: 'No Tumor',
    desc: 'A healthy brain scan with no signs of abnormal growths or masses.',
    confidence: '97.1%',
  },
  {
    label: 'Pituitary',
    desc: 'A growth on the pituitary gland at the base of the brain. Often small and can affect hormone levels.',
    confidence: '89.5%',
  },
]

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function Results() {
  const [samples, setSamples] = useState([])
  const [activeDemo, setActiveDemo] = useState(null)
  const [demoResult, setDemoResult] = useState(null)
  const [demoLoading, setDemoLoading] = useState(false)

  useEffect(() => {
    let active = true
    fetch('/samples.json')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('samples unavailable'))))
      .then((data) => { if (active && Array.isArray(data)) setSamples(data) })
      .catch(() => { if (active) setSamples([]) })
    return () => { active = false }
  }, [])

  const runLiveDemo = async (index) => {
    const sample = samples[index]
    if (!sample?.b64) return
    setActiveDemo(index)
    setDemoLoading(true)
    setDemoResult(null)

    try {
      const binary = atob(sample.b64)
      const bytes = new Uint8Array(binary.length)
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
      const blob = new Blob([bytes], { type: 'image/jpeg' })

      const formData = new FormData()
      formData.append('file', blob, 'sample.jpg')

      const res = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        setDemoResult({ error: `API returned status ${res.status}.` })
        return
      }

      const data = await res.json()
      const prediction = data?.prediction
      if (!prediction) {
        setDemoResult({ error: 'The server returned an unexpected response.' })
        return
      }

      setDemoResult({
        predictedClass: prediction.class ?? 'Unknown',
        confidence: prediction.confidence,
        uncertainty: prediction.uncertainty,
        probabilities: prediction.all_probabilities ?? {},
      })
    } catch {
      setDemoResult({
        error: 'Could not connect to the API. Make sure it is running on port 8000.',
      })
    } finally {
      setDemoLoading(false)
    }
  }

  const avgAccuracy = (testResults.reduce((s, r) => s + r.accuracy, 0) / testResults.length).toFixed(1)

  return (
    <section id="results" className="results-section">
      <div className="container">
        <div className="section-header">
          <span className="eyebrow">04 / Validation</span>
          <h2 className="section-title">How well does it work?</h2>
          <p className="section-subtitle">
            We trained the AI on over {trainingInfo.totalImages} brain scans from multiple hospitals,
            then tested it on 5 completely separate sets of images to make sure it works reliably.
          </p>
        </div>

        <div className="results-overview">
          <div className="overview-stat">
            <div className="overview-value mono">{avgAccuracy}%</div>
            <div className="overview-label">Average accuracy</div>
          </div>
          <div className="overview-stat">
            <div className="overview-value mono">5/5</div>
            <div className="overview-label">Tests passed</div>
          </div>
          <div className="overview-stat">
            <div className="overview-value mono">0</div>
            <div className="overview-label">Tests failed</div>
          </div>
          <div className="overview-stat">
            <div className="overview-value mono">{trainingInfo.totalImages}</div>
            <div className="overview-label">Scans used for training</div>
          </div>
        </div>

        <h3 className="subsection-title">Test results across 5 different image sets</h3>
        <p className="subsection-desc">
          Each test uses brain scans the AI has never seen before. Higher accuracy means the AI got more answers right.
        </p>
        <div className="results-grid">
          {testResults.map((r) => (
            <div className={`result-card card reticle tone-${r.tone}`} key={r.name}>
              <div className="result-header">
                <span className="result-name">{r.name}</span>
                <span className="result-badge">{r.type}</span>
              </div>
              <div className="result-accuracy mono">
                {r.accuracy.toFixed(1)}%
              </div>
              <div className="result-bar-track">
                <div className="result-bar-fill" style={{ width: `${r.accuracy}%` }} />
              </div>
              <p className="result-desc">{r.desc}</p>
              <p className="result-detail">{r.detail}</p>
              <div className="result-meta mono">
                <span>{r.samples} scans tested</span>
              </div>
            </div>
          ))}
        </div>

        <h3 className="subsection-title">What did the AI learn from?</h3>
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
                  <td className="mono">{s.count}</td>
                  <td><span className="modality-tag">{s.type}</span></td>
                </tr>
              ))}
              <tr className="total-row">
                <td><strong>Total (each tumor type equally represented)</strong></td>
                <td className="mono"><strong>{trainingInfo.totalImages}</strong></td>
                <td><span className="modality-tag">All MRI types</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="training-meta">
          <span>AI model: <strong>{trainingInfo.model}</strong></span>
          <span>Scan size: <strong>{trainingInfo.inputSize}</strong></span>
          <span>Training: <strong>{trainingInfo.phases}</strong></span>
        </div>

        <h3 className="subsection-title">Try it yourself — live examples</h3>
        <p className="subsection-desc">
          Click any brain scan below to send it to the AI right now. You'll see the prediction,
          how confident the AI is, and the probability breakdown for each tumor type — all in real time.
        </p>

        {samples.length === 0 && (
          <div className="examples-empty card">
            <p>Sample scans are loading, or the demo assets are unavailable in this environment.</p>
          </div>
        )}

        <div className="examples-grid">
          {liveExamples.map((ex, i) => (
            <div
              className={`example-card card reticle ${activeDemo === i ? 'active' : ''} ${samples[i] ? '' : 'disabled'}`}
              key={ex.label}
              onClick={() => runLiveDemo(i)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') runLiveDemo(i) }}
            >
              {samples[i]?.b64 && (
                <img
                  className="example-img"
                  src={`data:image/jpeg;base64,${samples[i].b64}`}
                  alt={`${ex.label} MRI sample`}
                />
              )}
              <div className="example-info">
                <div className="example-label">{ex.label}</div>
                <p className="example-desc">{ex.desc}</p>
                <div className="example-meta mono">
                  <span>Expected confidence: {ex.confidence}</span>
                </div>
              </div>
              {activeDemo === i && demoLoading && (
                <div className="demo-overlay">
                  <div className="demo-spinner" />
                  <span>Analyzing…</span>
                </div>
              )}
            </div>
          ))}
        </div>

        {demoResult && !demoResult.error && (
          <div className="demo-result card reticle">
            <h4>AI's analysis (live)</h4>
            <div className="demo-result-grid">
              <div className="demo-field">
                <span className="demo-field-label">Predicted finding</span>
                <span className="demo-field-value highlight">
                  {demoResult.predictedClass}
                </span>
              </div>
              <div className="demo-field">
                <span className="demo-field-label">Confidence</span>
                <span className="demo-field-value mono">
                  {demoResult.confidence != null ? `${demoResult.confidence}%` : '—'}
                </span>
              </div>
              <div className="demo-field">
                <span className="demo-field-label">Uncertainty</span>
                <span className="demo-field-value mono">
                  {demoResult.uncertainty != null ? `±${demoResult.uncertainty}%` : '—'}
                </span>
              </div>
            </div>
            {Object.keys(demoResult.probabilities || {}).length > 0 && (
              <div className="demo-probs">
                <span className="demo-field-label">Likelihood of each type</span>
                {Object.entries(demoResult.probabilities).map(([cls, prob]) => (
                  <div className="prob-row" key={cls}>
                    <span className="prob-name">{cls}</span>
                    <div className="prob-bar-track">
                      <div
                        className="prob-bar-fill"
                        style={{ width: `${Math.max(0, Math.min(100, Number(prob) || 0))}%` }}
                      />
                    </div>
                    <span className="prob-value mono">{prob}%</span>
                  </div>
                ))}
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
