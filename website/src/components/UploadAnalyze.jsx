import { useState } from 'react'
import './UploadAnalyze.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const cleanText = (text) =>
  typeof text === 'string' ? text.replace(/[^\x00-\x7F]+/g, '').trim() : ''

export default function UploadAnalyze() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragActive, setDragActive] = useState(false)

  const handleFile = (f) => {
    if (!f) return
    setFile(f)
    setResult(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.onerror = () => setError('Could not read the selected file. Please try a different scan.')
    reader.readAsDataURL(f)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragActive(false)
    handleFile(e.dataTransfer?.files?.[0])
  }

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        let detail = `Analysis failed (status ${res.status}).`
        try {
          const err = await res.json()
          if (err?.detail) detail = err.detail
        } catch {
          /* response had no JSON body */
        }
        throw new Error(detail)
      }

      const data = await res.json()
      if (!data || !data.prediction) {
        throw new Error('The server returned an unexpected response. Please try again.')
      }
      setResult(data)
    } catch (err) {
      setError(
        err?.message ||
          'Could not connect to the analysis server. Please ensure the API is running.'
      )
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  const prediction = result?.prediction
  const clinical = result?.clinical ?? {}
  const probabilities = prediction?.all_probabilities ?? {}
  const gradcam = result?.explainability?.gradcam_overlay
  const confidenceLevel = clinical.confidence_level || 'moderate'

  return (
    <section id="analyze">
      <div className="container">
        <div className="section-header">
          <span className="eyebrow">01 / Intake</span>
          <h2 className="section-title">Analyze MRI scan</h2>
          <p className="section-subtitle">
            Upload a brain MRI image to receive AI-assisted classification with visual explanation.
          </p>
        </div>

        <div className="analyze-layout">
          <div className="analyze-upload">
            {!preview ? (
              <div
                className={`upload-zone reticle ${dragActive ? 'active' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input').click()}
              >
                <svg width="52" height="52" viewBox="0 0 48 48" fill="none" aria-hidden="true">
                  <rect x="6" y="6" width="36" height="36" rx="3" stroke="var(--primary)" strokeWidth="1.6"/>
                  <path d="M24 16v16M16 24h16" stroke="var(--primary)" strokeWidth="1.6" strokeLinecap="round"/>
                  <path d="M6 16h4M6 32h4M38 16h4M38 32h4" stroke="var(--text-muted)" strokeWidth="1.4" strokeLinecap="round"/>
                </svg>
                <p className="upload-title">Drop MRI scan here</p>
                <p className="upload-hint">or click to browse files</p>
                <p className="upload-formats mono">JPG · PNG · BMP · TIFF · DICOM (.dcm)</p>
                <input
                  id="file-input"
                  type="file"
                  accept=".jpg,.jpeg,.png,.bmp,.tiff,.dcm,.dicom"
                  onChange={(e) => handleFile(e.target.files?.[0])}
                  hidden
                />
              </div>
            ) : (
              <div className="upload-preview reticle">
                <img src={preview} alt="MRI preview" className="preview-img" />
                <div className="preview-info">
                  <span className="preview-name">{file?.name || 'scan'}</span>
                  <span className="preview-size mono">
                    {file?.size ? `${(file.size / 1024).toFixed(0)} KB` : ''}
                  </span>
                </div>
                <div className="preview-actions">
                  <button className="btn-analyze" onClick={handleSubmit} disabled={loading}>
                    {loading ? (
                      <>
                        <span className="spinner"></span>
                        Analyzing…
                      </>
                    ) : (
                      'Run analysis'
                    )}
                  </button>
                  <button className="btn-reset" onClick={resetForm}>Choose different scan</button>
                </div>
              </div>
            )}
          </div>

          <div className="analyze-results reticle">
            {!result && !error && !loading && (
              <div className="results-placeholder">
                <svg width="60" height="60" viewBox="0 0 48 48" fill="none" aria-hidden="true">
                  <circle cx="24" cy="24" r="18" stroke="var(--border-strong)" strokeWidth="1.4"/>
                  <path d="M24 24V13M24 24l8 5" stroke="var(--border-strong)" strokeWidth="1.4" strokeLinecap="round"/>
                </svg>
                <p>Results will appear here after analysis</p>
              </div>
            )}

            {loading && (
              <div className="results-loading">
                <div className="loading-brain">
                  <span className="spinner-lg"></span>
                </div>
                <p className="loading-text">Analyzing scan…</p>
                <p className="loading-sub">Running classification, uncertainty check, and generating heatmap</p>
              </div>
            )}

            {error && (
              <div className="results-error">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <rect x="3" y="3" width="18" height="18" rx="2" stroke="var(--danger)" strokeWidth="1.6"/>
                  <path d="M12 8v5M12 16h.01" stroke="var(--danger)" strokeWidth="1.8" strokeLinecap="round"/>
                </svg>
                <p>{error}</p>
                <button className="btn-retry" onClick={handleSubmit} disabled={!file}>Retry</button>
              </div>
            )}

            {result && prediction && (
              <div className="results-content">
                <div className={`result-banner ${confidenceLevel}`}>
                  <div className="result-class">{prediction.class ?? 'Unknown'}</div>
                  <div className="result-confidence mono">
                    {prediction.confidence != null ? `${prediction.confidence}%` : '—'}
                  </div>
                </div>

                {clinical.recommendation && (
                  <div className={`result-recommendation ${clinical.flagged_for_review ? 'flagged' : ''}`}>
                    <p>{cleanText(clinical.recommendation)}</p>
                  </div>
                )}

                <div className="result-metrics">
                  <div className="metric">
                    <span className="metric-label">Confidence</span>
                    <span className="metric-value mono">
                      {prediction.confidence != null ? `${prediction.confidence}%` : '—'}
                    </span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Uncertainty</span>
                    <span className="metric-value mono">
                      {prediction.uncertainty != null ? `±${prediction.uncertainty}%` : '—'}
                    </span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Confidence level</span>
                    <span className={`metric-badge ${confidenceLevel}`}>
                      {confidenceLevel}
                    </span>
                  </div>
                </div>

                {Object.keys(probabilities).length > 0 && (
                  <div className="result-probs">
                    <h4>Classification probabilities</h4>
                    {Object.entries(probabilities).map(([cls, prob]) => (
                      <div className="prob-row" key={cls}>
                        <span className="prob-name">{cls}</span>
                        <div className="prob-bar-track">
                          <div
                            className={`prob-bar-fill ${cls === prediction.class ? 'primary' : ''}`}
                            style={{ width: `${Math.max(0, Math.min(100, Number(prob) || 0))}%` }}
                          ></div>
                        </div>
                        <span className="prob-val mono">{prob}%</span>
                      </div>
                    ))}
                  </div>
                )}

                {gradcam && (
                  <div className="result-gradcam">
                    <h4>Visual explanation — Grad-CAM++</h4>
                    <p className="gradcam-desc">
                      Highlighted regions show where the AI focused to make its assessment.
                      Warm colors indicate high importance.
                    </p>
                    <img
                      src={`data:image/png;base64,${gradcam}`}
                      alt="Grad-CAM++ heatmap overlay"
                      className="gradcam-img"
                    />
                  </div>
                )}

                <button className="btn-new-scan" onClick={resetForm}>Analyze another scan</button>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
