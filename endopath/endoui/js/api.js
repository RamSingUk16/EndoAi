const API = {
  async get(path) {
    const res = await fetch(path, { credentials: 'include' })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  },
  async post(path, bodyObj) {
    const res = await fetch(path, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(bodyObj),
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  },
  async postForm(path, formData) {
    const res = await fetch(path, {
      method: 'POST',
      credentials: 'include',
      body: formData,
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }
}
