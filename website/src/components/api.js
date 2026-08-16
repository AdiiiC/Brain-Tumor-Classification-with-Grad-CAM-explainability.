const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Deployments with API_KEYS configured require this header on every clinical endpoint.
const API_KEY = import.meta.env.VITE_API_KEY || ''

export { API_URL }

// Strips non-printable and non-ASCII characters (e.g. emoji) from server strings,
// keeping ordinary whitespace intact.
export const cleanText = (text) =>
  typeof text === 'string' ? text.replace(/[^\x20-\x7E\n\r\t]+/g, '').trim() : ''

const authHeaders = (extra = {}) =>
  API_KEY ? { 'X-API-Key': API_KEY, ...extra } : { ...extra }

const parseError = async (res, fallback) => {
  let detail = fallback ?? `Request failed (status ${res.status}).`
  try {
    const body = await res.json()
    if (body?.detail && typeof body.detail === 'string') detail = body.detail
  } catch {
    /* response had no JSON body */
  }
  if (res.status === 401 || res.status === 403) {
    detail = 'This deployment requires an API key. Set VITE_API_KEY in your environment.'
  }
  if (res.status === 429) {
    const retryAfter = res.headers.get('Retry-After')
    detail = `Too many requests. Please retry${retryAfter ? ` in ${retryAfter}s` : ' shortly'}.`
  }
  const err = new Error(detail)
  err.status = res.status
  return err
}

const buildUrl = (path, params) => {
  const url = new URL(`${API_URL}${path}`)
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        url.searchParams.set(key, String(value))
      }
    })
  }
  return url.toString()
}

export async function postFile(path, file, { params, field = 'file' } = {}) {
  const formData = new FormData()
  formData.append(field, file)

  const res = await fetch(buildUrl(path, params), {
    method: 'POST',
    body: formData,
    headers: authHeaders(),
  })
  if (!res.ok) throw await parseError(res)
  return res.json()
}

export async function postFiles(path, files, { field = 'files' } = {}) {
  const formData = new FormData()
  files.forEach((file) => formData.append(field, file))

  const res = await fetch(`${API_URL}${path}`, {
    method: 'POST',
    body: formData,
    headers: authHeaders(),
  })
  if (!res.ok) throw await parseError(res)
  return res.json()
}

export async function getJson(path, params) {
  const res = await fetch(buildUrl(path, params), { headers: authHeaders() })
  if (!res.ok) throw await parseError(res)
  return res.json()
}

export async function postJson(path, body) {
  const res = await fetch(`${API_URL}${path}`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(body),
  })
  if (!res.ok) throw await parseError(res)
  return res.json()
}

/** Fetch the PDF report for a study and trigger a browser download. */
export async function downloadReport(studyId) {
  const res = await fetch(`${API_URL}/studies/${studyId}/report`, { headers: authHeaders() })
  if (!res.ok) throw await parseError(res)

  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `brainscan-report-${studyId.slice(0, 8)}.pdf`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

export const submitFeedback = (studyId, payload) =>
  postJson(`/studies/${studyId}/feedback`, payload)

export const fetchTimeline = (patientId) =>
  getJson(`/patients/${encodeURIComponent(patientId)}/timeline`)

export const fetchStudies = (params) => getJson('/studies', params)
