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

    // collect metadata - updated to match backend API
    const fm = new FormData()
    fm.append('file', file)  // Backend expects 'file'
    fm.append('patient_id', form.specimen_id.value || 'UNKNOWN')  // Backend expects 'patient_id'
    
    // Map age_group to clinical_history since backend expects numeric age
    const historyParts = []
    if (form.age_group && form.age_group.value !== 'unknown') {
      historyParts.push(`Age group: ${form.age_group.value}`)
    }
    if (form.menstrual_phase && form.menstrual_phase.value !== 'unknown') {
      historyParts.push(`Menstrual phase: ${form.menstrual_phase.value}`)
    }
    if (form.magnification && form.magnification.value) {
      historyParts.push(`Magnification: ${form.magnification.value}`)
    }
    if (form.stain && form.stain.value) {
      historyParts.push(`Stain: ${form.stain.value}`)
    }
    if (form.notes && form.notes.value) {
      historyParts.push(form.notes.value)
    }
    if (historyParts.length > 0) {
      fm.append('clinical_history', historyParts.join('; '))
    }

    const gradcam = form.gradcam && form.gradcam.checked ? 'on' : 'auto'
    fm.append('gradcam', gradcam)
    
    // Debug: log what we're sending
    console.log('Uploading with FormData:')
    for (let pair of fm.entries()) {
      console.log(pair[0] + ': ' + (pair[1] instanceof File ? pair[1].name : pair[1]))
    }

    // disable UI while uploading
    uploadBtn.disabled = true
    uploadBtn.textContent = 'Uploading...'
    try{
      const json = await API.postForm('/cases', fm)
      const caseId = json.id
      const slideId = json.slide_id || caseId || 'unknown'
      showMessage('✅ Successfully uploaded! Slide ID: ' + slideId, 'success')
      
      // Redirect to case details page to show confirmation and results
      window.location.href = `case-detail.html?id=${caseId}`
    }catch(err){
      console.error('Upload error:', err)
      showMessage('Upload failed: '+err.message, 'error')
    }finally{
      uploadBtn.disabled = false
      uploadBtn.textContent = 'Upload'
    }
  })
})
