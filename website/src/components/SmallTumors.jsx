import { useRef, useState } from 'react'
import { postFile } from './api'
import './SmallTumors.css'

const SENSITIVITIES = ['low', 'medium', 'high']

export default function SmallTumors({ file, preview }) {
  const [sensitivity, setSensitivity] = useState('high')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [natural, setNatural] = useState(null)
  const imgRef = useRef(null)

  const detections = [
    ...(result?.small_lesions_detected ?? []),
    ...(result?.larger_lesions_detected ?? []),
  ]

  const parseResolution = (value) => {
    if (typeof value !== 'string') return null
    const [w, h] = value.toLowerCase().split('x').map((n) => Number(n))
    return w > 0 && h > 0 ? { w, h } : null
  }

  const dims = natural ?? parseResolution(result?.image_resolution)

  const boxStyle = (loc) => {
    if (!dims || !loc) return { display: 'none' }
    const { x = 0, y = 0, width = 0, height = 0 } = loc
    return {
      left: `${(x / dims.w) * 100}%`,
      top: `${(y / dims.h) * 100}%`,
      width: `${(width / dims.w) * 100}%`,
      height: `${(height / dims.h) * 100}%`,
    }
  }

  const runScan = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await postFile('/detect/small-tumors', file, {
        params: { sensitivity },
      })
      setResult(data)
    } catch (err) {
      setError(
        err?.status === 400
          ? 'Small-tumor detection needs the full Keras model, which is not active in this deployment.'
          : err?.message || 'Could not run small-tumor detection.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="small-tumors">
      <div className="st-header">
        <div>
          <h4>Small-lesion scan</h4>
          <p className="st-sub">
            Sliding-window search for sub-5&nbsp;mm lesions that whole-image classification can miss.
          </p>
        </div>
        <div className="st-controls">
          <label className="st-field">
            <span className="st-field-label mono">Sensitivity</span>
            <select
              className="st-select"
              value={sensitivity}
              onChange={(e) => setSensitivity(e.target.value)}
              disabled={loading}
            >
              {SENSITIVITIES.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </label>
          <button className="st-run" onClick={runScan} disabled={loading}>
            {loading ? 'Scanning…' : 'Scan for lesions'}
          </button>
        </div>
      </div>

      {error && <div className="st-error">{error}</div>}

      {result && (
        <>
          <div className="st-overlay-wrap">
            <img
              ref={imgRef}
              src={preview}
              alt="Scan with detected lesions"
              className="st-overlay-img"
              onLoad={(e) =>
                setNatural({ w: e.target.naturalWidth, h: e.target.naturalHeight })
              }
            />
            {detections.map((d, i) => (
              <span
                key={i}
                className={`st-box ${(d.estimated_size_mm ?? 0) < 5 ? 'small' : 'large'}`}
                style={boxStyle(d.location)}
              >
                <span className="st-box-tag mono">
                  {(d.tumor_probability != null
                    ? Math.round(d.tumor_probability * 100)
                    : '?')}%
                </span>
              </span>
            ))}
          </div>

          <div className="st-summary mono">
            <span>{result.total_suspicious_regions ?? detections.length} suspicious region(s)</span>
            <span>{result.total_patches_analyzed ?? '—'} patches analyzed</span>
            <span>{result.classifier ?? '—'}</span>
          </div>

          {result.clinical_note && <p className="st-note">{result.clinical_note}</p>}
          {result.note && <p className="st-note warn">{result.note}</p>}

          {detections.length > 0 && (
            <div className="st-list">
              {detections.map((d, i) => (
                <div className="st-item" key={i}>
                  <span className="st-item-idx mono">#{i + 1}</span>
                  <span className="st-item-class">{d.dominant_class || 'Suspicious tissue'}</span>
                  <span className="st-item-meta mono">
                    {d.estimated_size_mm != null ? `${d.estimated_size_mm.toFixed(1)} mm` : '— mm'}
                  </span>
                  <span className="st-item-prob mono">
                    {d.tumor_probability != null
                      ? `${(d.tumor_probability * 100).toFixed(0)}%`
                      : '—'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}
