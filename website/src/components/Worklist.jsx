import { useMemo, useState } from 'react'
import { postFiles } from './api'
import './Worklist.css'

const MAX_FILES = 100

export default function Worklist() {
  const [files, setFiles] = useState([])
  const [rows, setRows] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const pickFiles = (list) => {
    const picked = Array.from(list || []).slice(0, MAX_FILES)
    setFiles(picked)
    setRows(null)
    setError(null)
  }

  const run = async () => {
    if (files.length === 0) return
    setLoading(true)
    setError(null)
    setRows(null)
    try {
      const data = await postFiles('/predict/batch', files)
      if (!Array.isArray(data)) throw new Error('Unexpected response from the batch endpoint.')
      setRows(data)
    } catch (err) {
      setError(err?.message || 'Batch analysis failed. Please ensure the API is running.')
    } finally {
      setLoading(false)
    }
  }

  const sorted = useMemo(() => {
    if (!rows) return []
    const priority = (row) => {
      const cls = row?.result?.predicted_class
      if (cls === 'Error') return 0
      if (row?.result?.flagged_for_review) return 1
      return 2
    }
    return [...rows].sort((a, b) => priority(a) - priority(b))
  }, [rows])

  const flaggedCount = sorted.filter((r) => r?.result?.flagged_for_review).length
  const errorCount = sorted.filter((r) => r?.result?.predicted_class === 'Error').length

  return (
    <section id="worklist">
      <div className="container">
        <div className="section-header">
          <span className="eyebrow">06 / Worklist</span>
          <h2 className="section-title">Batch triage</h2>
          <p className="section-subtitle">
            Queue up to {MAX_FILES} scans for rapid classification. Flagged and failed studies
            surface at the top for review.
          </p>
        </div>

        <div className="wl-panel reticle">
          <div className="wl-controls">
            <label className="wl-picker">
              <input
                type="file"
                multiple
                accept=".jpg,.jpeg,.png,.bmp,.tiff,.dcm,.dicom"
                onChange={(e) => pickFiles(e.target.files)}
                hidden
              />
              <span className="wl-picker-btn">Select scans</span>
              <span className="wl-picker-count mono">
                {files.length > 0 ? `${files.length} selected` : 'no files selected'}
              </span>
            </label>
            <button className="wl-run" onClick={run} disabled={loading || files.length === 0}>
              {loading ? 'Processing…' : 'Run batch'}
            </button>
          </div>

          {error && <div className="wl-error">{error}</div>}

          {loading && <div className="wl-status">Classifying {files.length} scan(s)…</div>}

          {sorted.length > 0 && (
            <>
              <div className="wl-summary mono">
                <span>{sorted.length} studies</span>
                <span className="wl-flag">{flaggedCount} flagged</span>
                {errorCount > 0 && <span className="wl-err">{errorCount} errors</span>}
              </div>

              <div className="wl-table">
                <div className="wl-head">
                  <span>Study</span>
                  <span>Prediction</span>
                  <span className="wl-num">Confidence</span>
                  <span className="wl-num">Status</span>
                </div>
                {sorted.map((row, i) => {
                  const res = row?.result ?? {}
                  const isError = res.predicted_class === 'Error'
                  const conf = res.confidence
                  return (
                    <div
                      className={`wl-row ${isError ? 'error' : res.flagged_for_review ? 'flagged' : ''}`}
                      key={`${row?.filename || 'file'}-${i}`}
                    >
                      <span className="wl-file" title={row?.filename}>{row?.filename || '—'}</span>
                      <span className="wl-pred">{res.predicted_class ?? '—'}</span>
                      <span className="wl-num mono">
                        {conf != null ? `${(Number(conf) * 100).toFixed(1)}%` : '—'}
                      </span>
                      <span className="wl-num">
                        {isError ? (
                          <span className="wl-tag err">error</span>
                        ) : res.flagged_for_review ? (
                          <span className="wl-tag flag">review</span>
                        ) : (
                          <span className="wl-tag ok">clear</span>
                        )}
                      </span>
                    </div>
                  )
                })}
              </div>
            </>
          )}

          {rows && sorted.length === 0 && !loading && (
            <div className="wl-status">No results returned.</div>
          )}
        </div>
      </div>
    </section>
  )
}
