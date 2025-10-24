document.addEventListener('DOMContentLoaded', ()=>{
  const form = document.getElementById('uploadForm')
  if (!form) return
  form.addEventListener('submit', async (e)=>{
    e.preventDefault()
    const f = form.querySelector('input[type=file]')
    const file = f.files[0]
    if (!file) return showMessage('Please choose a file', 'error')
    if (file.size > 10*1024*1024) return showMessage('File too large (max 10MB)', 'error')
    const fm = new FormData()
    fm.append('image', file)
    fm.append('specimen_id', form.specimen_id.value || '')
    fm.append('notes', form.notes.value || '')
    try{
      const json = await API.postForm('/cases', fm)
      showMessage('Uploaded: Slide ID '+(json.id||json.slide||'unknown'), 'success')
      document.getElementById('result').innerHTML = `<a href="results.html">View results</a>`
    }catch(err){
      console.error(err)
      showMessage('Upload failed: '+err.message, 'error')
    }
  })
})
