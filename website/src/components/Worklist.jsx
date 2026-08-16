import { useCallback, useMemo, useState } from 'react'
import { postFiles, fetchStudies, downloadReport } from './api'
import './Worklist.css'

const MAX_FILES = 100

export default function Worklist() {
  const [tab, setTab] = useState('batch')

  const [files, setFiles] = useState([])
  const [rows, setRows] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const [queue, setQueue] = useState([])
  const [queueLoading, setQueueLoading] = useState(false)
  const [queueError, setQueueError] = useState(null)
  const [flaggedOnly, setFlaggedOnly] = useState(true)

  const loadQueue = useCallback(async (onlyFlagged = flaggedOnly) => {
    setQueueLoading(true)
    setQueueError(null)
    try {
      const data = await fetchStudies({ flagged_only: onlyFlagged, limit: 100 })
      setQueue(Array.isArray(data) ? data : [])
    } catch (err) {
      setQueueError(err?.message || 'Could not load the review queue.')
    } finally {
      setQueueLoading(false)
    }
  }, [flaggedOnly])

  const openQueue = () => {
    setTab('queue')
    loadQueue()
  }

  const toggleFlaggedOnly = (checked) => {
    setFlaggedOnly(checked)
    loadQueue(checked)
  }

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
            Queue up to {MAX_FILES} scans for rapid classification, or work through the
            persistent review queue of studies the model flagged.
          </p>
        </div>

        <div className="wl-tabs">
          <button
            className={`wl-tab ${tab === 'batch' ? 'active' : ''}`}
            onClick={() => setTab('batch')}
          >
            Batch triage
          </button>
          <button
            className={`wl-tab ${tab === 'queue' ? 'active' : ''}`}
            onClick={openQueue}
          >
            Review queue
          </button>
        </div>

        {tab === 'queue' && (
          <div className="wl-panel reticle">
            <div className="wl-controls">
              <label className="wl-toggle">
                <input
                  type="checkbox"
                  checked={flaggedOnly}
                  onChange={(e) => toggleFlaggedOnly(e.target.checked)}
                />
                <span className="mono">Flagged only</span>
              </label>
              <button className="wl-run" onClick={() => loadQueue()} disabled={queueLoading}>
                {queueLoading ? 'Loading…' : 'Refresh'}
              </button>
            </div>

            {queueError && <div className="wl-error">{queueError}</div>}

            {!queueLoading && !queueError && queue.length === 0 && (
              <div className="wl-status">
                {flaggedOnly
                  ? 'Nothing awaiting review — no studies have been flagged.'
                  : 'No studies stored yet. Run an analysis to populate the queue.'}
              </div>
            )}

            {queue.length > 0 && (
              <>
                <div className="wl-summary mono">
                  <span>{queue.length} studies</span>
                  <span className="wl-flag">
                    {queue.filter((s) => s.confirmed_class == null).length} unreviewed
                  </span>
                </div>

                <div className="wl-table queue">
                  <div className="wl-head">
                    <span>Study</span>
                    <span>Prediction</span>
                    <span className="wl-num">Confidence</span>
                    <span className="wl-num">Status</span>
                    <span className="wl-num">Report</span>
                  </div>
                  {queue.map((s) => (
                    <div
                      className={`wl-row ${s.is_ood ? 'error' : s.flagged_for_review ? 'flagged' : ''}`}
                      key={s.id}
                    >
                      <span className="wl-file" title={s.filename || s.id}>
                        {s.patient_id ? `${s.patient_id} · ` : ''}
                        {s.filename || s.id.slice(0, 8)}
                        <span className="wl-date mono">
                          {new Date(s.created_at).toLocaleDateString()}
                        </span>
                      </span>
                      <span className="wl-pred">
                        {s.predicted_class}
                        {s.confirmed_class && s.confirmed_class !== s.predicted_class && (
                          <span className="wl-tag err"> → {s.confirmed_class}</span>
                        )}
                      </span>
                      <span className="wl-num mono">
                        {s.confidence != null ? `${(s.confidence * 100).toFixed(1)}%` : '—'}
                      </span>
                      <span className="wl-num">
                        {s.is_ood ? (
                          <span className="wl-tag err">out-of-dist</span>
                        ) : s.confirmed_class ? (
                          <span className="wl-tag ok">reviewed</span>
                        ) : s.flagged_for_review ? (
                          <span className="wl-tag flag">review</span>
                        ) : (
                          <span className="wl-tag ok">clear</span>
                        )}
                      </span>
                      <span className="wl-num">
                        <button className="wl-link" onClick={() => downloadReport(s.id)}>
                          PDF
                        </button>
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {tab === 'batch' && (
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
        )}
      </div>
    </section>
  )
}
