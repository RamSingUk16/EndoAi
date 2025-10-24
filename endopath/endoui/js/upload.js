document.addEventListener('DOMContentLoaded', ()=>{
  const form = document.getElementById('uploadForm')
  if (!form) return
  form.addEventListener('submit', async (e)=>{
    e.preventDefault()
    const uploadBtn = document.getElementById('uploadBtn')
    const f = form.querySelector('input[type=file]')
    const file = f.files[0]
    if (!file) return showMessage('Please choose a file', 'error')
    if (file.size > 10*1024*1024) return showMessage('File too large (max 10MB)', 'error')

    // collect metadata
    const fm = new FormData()
    fm.append('image', file)
    fm.append('specimen_id', form.specimen_id.value || '')
    fm.append('age_group', form.age_group.value || 'unknown')
    fm.append('menstrual_phase', form.menstrual_phase.value || 'unknown')
    fm.append('magnification', form.magnification.value || '')
    fm.append('stain', form.stain.value || '')
    fm.append('notes', form.notes.value || '')

    const gradcam = form.gradcam && form.gradcam.checked

    // disable UI while uploading
    uploadBtn.disabled = true
    uploadBtn.textContent = 'Uploading...'
    try{
      const path = '/cases' + (gradcam ? '?gradcam=auto' : '')
      const json = await API.postForm(path, fm)
      const id = json.id || json.slide || json.case_id || 'unknown'
      showMessage('Uploaded: Slide ID '+id, 'success')
      document.getElementById('result').innerHTML = `<a href="results.html">View results</a> — Slide ${id}`
    }catch(err){
      console.error(err)
      showMessage('Upload failed: '+err.message, 'error')
    }finally{
      uploadBtn.disabled = false
      uploadBtn.textContent = 'Upload'
    }
  })
})
