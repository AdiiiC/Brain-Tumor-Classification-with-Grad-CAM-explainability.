import { useState } from 'react'
import './UploadAnalyze.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

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
    reader.readAsDataURL(f)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragActive(false)
    handleFile(e.dataTransfer.files[0])
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
        const err = await res.json()
        throw new Error(err.detail || 'Analysis failed')
      }

      setResult(await res.json())
    } catch (err) {
      setError(err.message || 'Could not connect to the analysis server. Please ensure the API is running.')
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

  return (
    <section id="analyze">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">Analyze MRI Scan</h2>
          <p className="section-subtitle">
            Upload a brain MRI image to receive AI-assisted classification with visual explanation.
          </p>
        </div>

        <div className="analyze-layout">
          {/* Upload Area */}
          <div className="analyze-upload">
            {!preview ? (
              <div
                className={`upload-zone ${dragActive ? 'active' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input').click()}
              >
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
                </svg>
                <p className="upload-title">Drop MRI scan here</p>
                <p className="upload-hint">or click to browse files</p>
                <p className="upload-formats">JPG, PNG, BMP, TIFF, DICOM (.dcm)</p>
                <input
                  id="file-input"
                  type="file"
                  accept=".jpg,.jpeg,.png,.bmp,.tiff,.dcm,.dicom"
                  onChange={(e) => handleFile(e.target.files[0])}
                  hidden
                />
              </div>
            ) : (
              <div className="upload-preview">
                <img src={preview} alt="MRI Preview" className="preview-img" />
                <div className="preview-info">
                  <span className="preview-name">{file.name}</span>
                  <span className="preview-size">{(file.size / 1024).toFixed(0)} KB</span>
                </div>
                <div className="preview-actions">
                  <button className="btn-analyze" onClick={handleSubmit} disabled={loading}>
                    {loading ? (
                      <>
                        <span className="spinner"></span>
                        Analyzing...
                      </>
                    ) : (
                      'Run Analysis'
                    )}
                  </button>
                  <button className="btn-reset" onClick={resetForm}>Choose Different Scan</button>
                </div>
              </div>
            )}
          </div>

          {/* Results Area */}
          <div className="analyze-results">
            {!result && !error && !loading && (
              <div className="results-placeholder">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="var(--border)" strokeWidth="1">
                  <circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>
                </svg>
                <p>Results will appear here after analysis</p>
              </div>
            )}

            {loading && (
              <div className="results-loading">
                <div className="loading-brain">
                  <span className="spinner-lg"></span>
                </div>
                <p className="loading-text">Analyzing scan...</p>
                <p className="loading-sub">Running classification, uncertainty check, and generating heatmap</p>
              </div>
            )}

            {error && (
              <div className="results-error">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--danger)" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/>
                </svg>
                <p>{error}</p>
                <button className="btn-retry" onClick={handleSubmit}>Retry</button>
              </div>
            )}

            {result && (
              <div className="results-content">
                {/* Prediction */}
                <div className={`result-banner ${result.clinical.confidence_level}`}>
                  <div className="result-class">{result.prediction.class}</div>
                  <div className="result-confidence">{result.prediction.confidence}%</div>
                </div>

                {/* Clinical recommendation */}
                <div className={`result-recommendation ${result.clinical.flagged_for_review ? 'flagged' : ''}`}>
                  <p>{result.clinical.recommendation}</p>
                </div>

                {/* Metrics */}
                <div className="result-metrics">
                  <div className="metric">
                    <span className="metric-label">Confidence</span>
                    <span className="metric-value">{result.prediction.confidence}%</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Uncertainty</span>
                    <span className="metric-value">±{result.prediction.uncertainty}%</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Confidence Level</span>
                    <span className={`metric-badge ${result.clinical.confidence_level}`}>
                      {result.clinical.confidence_level}
                    </span>
                  </div>
                </div>

                {/* Probability breakdown */}
                <div className="result-probs">
                  <h4>Classification Probabilities</h4>
                  {Object.entries(result.prediction.all_probabilities).map(([cls, prob]) => (
                    <div className="prob-row" key={cls}>
                      <span className="prob-name">{cls}</span>
                      <div className="prob-bar-track">
                        <div
                          className={`prob-bar-fill ${cls === result.prediction.class ? 'primary' : ''}`}
                          style={{ width: `${prob}%` }}
                        ></div>
                      </div>
                      <span className="prob-val">{prob}%</span>
                    </div>
                  ))}
                </div>

                {/* Grad-CAM overlay */}
                {result.explainability.gradcam_overlay && (
                  <div className="result-gradcam">
                    <h4>Visual Explanation (Grad-CAM++)</h4>
                    <p className="gradcam-desc">
                      Highlighted regions show where the AI focused to make its assessment.
                      Warm colors (red/yellow) indicate high importance.
                    </p>
                    <img
                      src={`data:image/png;base64,${result.explainability.gradcam_overlay}`}
                      alt="Grad-CAM++ heatmap overlay"
                      className="gradcam-img"
                    />
                  </div>
                )}

                <button className="btn-new-scan" onClick={resetForm}>Analyze Another Scan</button>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
