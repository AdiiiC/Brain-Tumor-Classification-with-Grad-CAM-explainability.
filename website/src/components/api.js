const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export { API_URL }

export const cleanText = (text) =>
  typeof text === 'string' ? text.replace(/[^\x00-\x7F]+/g, '').trim() : ''

const parseError = async (res, fallback) => {
  let detail = fallback ?? `Request failed (status ${res.status}).`
  try {
    const body = await res.json()
    if (body?.detail && typeof body.detail === 'string') detail = body.detail
  } catch {
    /* response had no JSON body */
  }
  const err = new Error(detail)
  err.status = res.status
  return err
}

export async function postFile(path, file, { params, field = 'file' } = {}) {
  const url = new URL(`${API_URL}${path}`)
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        url.searchParams.set(key, String(value))
      }
    })
  }

  const formData = new FormData()
  formData.append(field, file)

  const res = await fetch(url.toString(), { method: 'POST', body: formData })
  if (!res.ok) throw await parseError(res)
  return res.json()
}

export async function postFiles(path, files, { field = 'files' } = {}) {
  const formData = new FormData()
  files.forEach((file) => formData.append(field, file))

  const res = await fetch(`${API_URL}${path}`, { method: 'POST', body: formData })
  if (!res.ok) throw await parseError(res)
  return res.json()
}
