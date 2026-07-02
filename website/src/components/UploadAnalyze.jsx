import { useState } from 'react'
import { postFile, cleanText } from './api'
import SmallTumors from './SmallTumors'
import './UploadAnalyze.css'

const affinityTone = { high: 'high', moderate: 'moderate', low: 'low', unknown: 'muted' }

export default function UploadAnalyze() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragActive, setDragActive] = useState(false)

  const [fullMode, setFullMode] = useState(false)
  const [patientAge, setPatientAge] = useState('')
  const [patientSex, setPatientSex] = useState('')

  const [explainTab, setExplainTab] = useState('gradcam')
  const [shap, setShap] = useState({ loading: false, image: null, error: null })

  const handleFile = (f) => {
    if (!f) return
    setFile(f)
    setResult(null)
    setError(null)
    setShap({ loading: false, image: null, error: null })
    setExplainTab('gradcam')
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
    setShap({ loading: false, image: null, error: null })
    setExplainTab('gradcam')

    try {
      const path = fullMode ? '/analyze/comprehensive' : '/analyze'
      const params = fullMode
        ? { patient_age: patientAge, patient_sex: patientSex }
        : undefined
      const data = await postFile(path, file, { params })
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

  const loadShap = async () => {
    setExplainTab('shap')
    if (shap.image || shap.loading) return
    setShap({ loading: true, image: null, error: null })
    try {
      const data = await postFile('/explain/shap', file)
      if (!data?.shap_image_base64) throw new Error('No SHAP attribution returned.')
      setShap({ loading: false, image: data.shap_image_base64, error: null })
    } catch (err) {
      const unavailable = err?.status === 400 || err?.status === 501
      setShap({
        loading: false,
        image: null,
        error: unavailable
          ? 'SHAP attribution is not available in this deployment.'
          : err?.message || 'Could not generate SHAP attribution.',
      })
    }
  }

  const resetForm = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    setShap({ loading: false, image: null, error: null })
    setExplainTab('gradcam')
  }

  const prediction = result?.prediction
  const clinical = result?.clinical ?? {}
  const probabilities = prediction?.all_probabilities ?? {}
  const gradcam = result?.explainability?.gradcam_overlay
  const confidenceLevel = clinical.confidence_level || 'moderate'

  const quality = result?.image_quality
  const sequence = result?.sequence
  const grading = result?.grading
  const pediatric = result?.pediatric
  const dicom = result?.dicom_metadata
  const lowQuality = quality?.overall_score != null && quality.overall_score < 50
  const clamp = (n) => Math.max(0, Math.min(100, Number(n) || 0))

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

                <div className="analysis-mode">
                  <label className="mode-toggle">
                    <input
                      type="checkbox"
                      checked={fullMode}
                      onChange={(e) => setFullMode(e.target.checked)}
                    />
                    <span className="mode-switch" aria-hidden="true"></span>
                    <span className="mode-text">
                      <span className="mode-title">Full clinical analysis</span>
                      <span className="mode-desc">Adds quality, sequence, grade &amp; pediatric checks</span>
                    </span>
                  </label>

                  {fullMode && (
                    <div className="patient-fields">
                      <label className="patient-field">
                        <span className="mono">Age</span>
                        <input
                          type="number"
                          min="0"
                          max="120"
                          placeholder="optional"
                          value={patientAge}
                          onChange={(e) => setPatientAge(e.target.value)}
                        />
                      </label>
                      <label className="patient-field">
                        <span className="mono">Sex</span>
                        <select value={patientSex} onChange={(e) => setPatientSex(e.target.value)}>
                          <option value="">—</option>
                          <option value="M">M</option>
                          <option value="F">F</option>
                        </select>
                      </label>
                    </div>
                  )}
                </div>

                <div className="preview-actions">
                  <button className="btn-analyze" onClick={handleSubmit} disabled={loading}>
                    {loading ? (
                      <>
                        <span className="spinner"></span>
                        Analyzing…
                      </>
                    ) : (
                      fullMode ? 'Run full analysis' : 'Run analysis'
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
                <p className="loading-sub">
                  {fullMode
                    ? 'Quality, sequence, classification, grade, uncertainty and heatmap'
                    : 'Running classification, uncertainty check, and generating heatmap'}
                </p>
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
                {lowQuality && (
                  <div className="quality-alert">
                    <strong>Low image quality (score {quality.overall_score}).</strong>{' '}
                    Results may be unreliable — consider re-acquiring the scan.
                  </div>
                )}

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

                {(sequence || quality) && (
                  <div className="chip-row">
                    {sequence?.detected_sequence && (
                      <div className="info-chip">
                        <span className="chip-key mono">Sequence</span>
                        <span className="chip-val">{sequence.detected_sequence}</span>
                        {sequence.model_affinity && (
                          <span className={`chip-tag ${affinityTone[sequence.model_affinity] || 'muted'}`}>
                            {sequence.model_affinity} affinity
                          </span>
                        )}
                      </div>
                    )}
                    {quality?.overall_score != null && (
                      <div className="info-chip">
                        <span className="chip-key mono">Quality</span>
                        <span className="chip-val mono">{quality.overall_score}/100</span>
                        <span className={`chip-tag ${quality.pass ? 'high' : 'low'}`}>
                          {quality.pass ? 'pass' : 'review'}
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {Object.keys(probabilities).length > 0 && (
                  <div className="result-probs">
                    <h4>Classification probabilities</h4>
                    {Object.entries(probabilities).map(([cls, prob]) => (
                      <div className="prob-row" key={cls}>
                        <span className="prob-name">{cls}</span>
                        <div className="prob-bar-track">
                          <div
                            className={`prob-bar-fill ${cls === prediction.class ? 'primary' : ''}`}
                            style={{ width: `${clamp(prob)}%` }}
                          ></div>
                        </div>
                        <span className="prob-val mono">{prob}%</span>
                      </div>
                    ))}
                  </div>
                )}

                {sequence && (
                  <div className="detail-block">
                    <h4>MRI sequence</h4>
                    <p className="detail-lead">
                      {sequence.sequence_name || sequence.detected_sequence}
                      {sequence.confidence != null && (
                        <span className="detail-muted mono"> · {Math.round(sequence.confidence * 100)}% conf</span>
                      )}
                    </p>
                    {sequence.accuracy_note && <p className="detail-note">{sequence.accuracy_note}</p>}
                    {sequence.preprocessing_advice && (
                      <p className="detail-note">Preprocessing: {sequence.preprocessing_advice}</p>
                    )}
                  </div>
                )}

                {quality && (
                  <div className="detail-block">
                    <h4>Image quality</h4>
                    <div className="qm-grid">
                      {quality.metrics?.resolution && (
                        <div className="qm"><span className="mono">Resolution</span><b>{quality.metrics.resolution}</b></div>
                      )}
                      {quality.metrics?.sharpness != null && (
                        <div className="qm"><span className="mono">Sharpness</span><b className="mono">{Number(quality.metrics.sharpness).toFixed(1)}</b></div>
                      )}
                      {quality.metrics?.snr != null && (
                        <div className="qm"><span className="mono">SNR</span><b className="mono">{Number(quality.metrics.snr).toFixed(1)}</b></div>
                      )}
                      {quality.metrics?.brain_coverage != null && (
                        <div className="qm"><span className="mono">Brain coverage</span><b className="mono">{Math.round(quality.metrics.brain_coverage * 100)}%</b></div>
                      )}
                    </div>
                    {Array.isArray(quality.issues) && quality.issues.length > 0 && (
                      <ul className="detail-list warn">
                        {quality.issues.map((issue, i) => <li key={i}>{issue}</li>)}
                      </ul>
                    )}
                    {Array.isArray(quality.recommendations) && quality.recommendations.length > 0 && (
                      <ul className="detail-list">
                        {quality.recommendations.map((rec, i) => <li key={i}>{rec}</li>)}
                      </ul>
                    )}
                  </div>
                )}

                {grading && (grading.grade || grading.grade_name) && (
                  <div className="detail-block">
                    <h4>WHO grade estimate</h4>
                    {grading.grade == null ? (
                      <p className="detail-note">Not applicable — no tumor detected.</p>
                    ) : (
                      <>
                        <div className="grade-head">
                          <span className="grade-badge">{grading.grade_name || `Grade ${grading.grade}`}</span>
                          {grading.grade_confidence != null && (
                            <span className="detail-muted mono">{Math.round(grading.grade_confidence * 100)}% conf</span>
                          )}
                        </div>
                        {grading.grade_description && <p className="detail-lead">{grading.grade_description}</p>}
                        {grading.prognosis && <p className="detail-note"><b>Prognosis:</b> {grading.prognosis}</p>}
                        {grading.typical_treatment && <p className="detail-note"><b>Typical treatment:</b> {grading.typical_treatment}</p>}
                        {Array.isArray(grading.supporting_evidence) && grading.supporting_evidence.length > 0 && (
                          <ul className="detail-list">
                            {grading.supporting_evidence.map((ev, i) => <li key={i}>{ev}</li>)}
                          </ul>
                        )}
                        {grading.image_features?.enhancement_pattern && (
                          <p className="detail-muted mono">
                            enhancement: {grading.image_features.enhancement_pattern}
                            {grading.image_features.heterogeneity != null &&
                              ` · heterogeneity ${Math.round(grading.image_features.heterogeneity * 100)}%`}
                          </p>
                        )}
                        {grading.disclaimer && <p className="detail-disclaimer">{grading.disclaimer}</p>}
                      </>
                    )}
                  </div>
                )}

                {pediatric?.is_pediatric && (
                  <div className="detail-block pediatric">
                    <h4>Pediatric assessment</h4>
                    {pediatric.age_group && (
                      <p className="detail-lead">
                        {pediatric.age_group}
                        {pediatric.prediction_changed && (
                          <span className="chip-tag low"> adjusted prediction</span>
                        )}
                      </p>
                    )}
                    {pediatric.adjusted_probabilities && (
                      <div className="result-probs compact">
                        {Object.entries(pediatric.adjusted_probabilities).map(([cls, prob]) => (
                          <div className="prob-row" key={cls}>
                            <span className="prob-name">{cls}</span>
                            <div className="prob-bar-track">
                              <div className="prob-bar-fill" style={{ width: `${clamp(prob * 100)}%` }}></div>
                            </div>
                            <span className="prob-val mono">{(prob * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {pediatric.reliability_warning && (
                      <p className="detail-note warn">{cleanText(pediatric.reliability_warning)}</p>
                    )}
                    {Array.isArray(pediatric.differential_diagnoses) && pediatric.differential_diagnoses.length > 0 && (
                      <div className="tag-line">
                        <span className="mono">Differentials:</span>
                        {pediatric.differential_diagnoses.map((d, i) => (
                          <span className="pill-tag" key={i}>{d}</span>
                        ))}
                      </div>
                    )}
                    {Array.isArray(pediatric.recommended_workup) && pediatric.recommended_workup.length > 0 && (
                      <ul className="detail-list">
                        {pediatric.recommended_workup.map((w, i) => <li key={i}>{w}</li>)}
                      </ul>
                    )}
                  </div>
                )}

                {dicom && (
                  <div className="detail-block">
                    <h4>DICOM metadata</h4>
                    <div className="dicom-grid">
                      {[
                        ['Patient', dicom.patient_name],
                        ['ID', dicom.patient_id],
                        ['Study date', dicom.study_date],
                        ['Modality', dicom.modality],
                        ['Body part', dicom.body_part],
                        ['Slice thickness', dicom.slice_thickness != null ? `${dicom.slice_thickness} mm` : null],
                        ['Matrix', dicom.rows && dicom.columns ? `${dicom.rows} × ${dicom.columns}` : null],
                      ].map(([k, v]) => (
                        <div className="dicom-cell" key={k}>
                          <span className="mono">{k}</span>
                          <b>{v ?? '—'}</b>
                        </div>
                      ))}
                    </div>
                    <p className="detail-disclaimer">May contain protected health information (PHI).</p>
                  </div>
                )}

                {gradcam && (
                  <div className="result-gradcam">
                    <div className="explain-tabs">
                      <button
                        className={`explain-tab ${explainTab === 'gradcam' ? 'active' : ''}`}
                        onClick={() => setExplainTab('gradcam')}
                      >
                        Grad-CAM++
                      </button>
                      <button
                        className={`explain-tab ${explainTab === 'shap' ? 'active' : ''}`}
                        onClick={loadShap}
                      >
                        SHAP
                      </button>
                    </div>

                    {explainTab === 'gradcam' && (
                      <>
                        <p className="gradcam-desc">
                          Highlighted regions show where the AI focused. Warm colors indicate high importance.
                        </p>
                        <img
                          src={`data:image/png;base64,${gradcam}`}
                          alt="Grad-CAM++ heatmap overlay"
                          className="gradcam-img"
                        />
                      </>
                    )}

                    {explainTab === 'shap' && (
                      <>
                        <p className="gradcam-desc">
                          SHAP attributes each region as supporting or opposing the prediction.
                        </p>
                        {shap.loading && <div className="shap-status">Generating SHAP attribution…</div>}
                        {shap.error && <div className="shap-status error">{shap.error}</div>}
                        {shap.image && (
                          <img
                            src={`data:image/png;base64,${shap.image}`}
                            alt="SHAP attribution map"
                            className="gradcam-img"
                          />
                        )}
                      </>
                    )}
                  </div>
                )}

                {file && <SmallTumors file={file} preview={preview} />}

                <button className="btn-new-scan" onClick={resetForm}>Analyze another scan</button>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
