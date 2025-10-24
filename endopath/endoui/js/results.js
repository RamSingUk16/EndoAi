document.addEventListener('DOMContentLoaded', ()=>{
  const cards = document.getElementById('cards')
  const summary = document.getElementById('summary')
  async function load(){
    try{
      const data = await API.get('/cases')
      console.log('Loaded cases:', data)
      render(data.cases || data)  // Handle both {cases: [...]} and plain array
    }catch(err){
      console.error(err)
      showMessage('Failed to load results: '+err.message,'error')
    }
  }
  function render(list){
    if (!Array.isArray(list)) {
      console.error('Expected array but got:', list)
      return
    }
    summary.textContent = `Total: ${list.length}`
    cards.innerHTML = ''
    if (list.length === 0) {
      cards.innerHTML = '<div class="card">No cases uploaded yet. <a href="upload.html">Upload a case</a></div>'
      return
    }
    list.forEach(item=>{
      const c = document.createElement('div')
      c.className = 'card'
      
      const slideId = item.slide_id || item.id
      const status = item.status || 'unknown'
      const prediction = item.prediction || 'pending'
      const confidence = item.confidence ? (item.confidence * 100).toFixed(1) + '%' : 'N/A'
      const uploadedAt = item.uploaded_at ? new Date(item.uploaded_at).toLocaleString() : 'N/A'
      
      c.innerHTML = `
        <strong>Slide ID: ${slideId}</strong>
        <div>Patient ID: ${item.patient_id || 'N/A'}</div>
        <div>Status: <span class="status-${status}">${status}</span></div>
        <div>Prediction: ${prediction}</div>
        <div>Confidence: ${confidence}</div>
        <div>Uploaded: ${uploadedAt}</div>
        <div style="margin-top: 0.5rem;">
          <a href="case-detail.html?id=${item.id}" style="margin-right: 1rem;">View Details</a>
          ${item.gradcam_data || status === 'completed' ? `<a href="/cases/${item.id}/gradcam" target="_blank">View GradCAM</a>` : ''}
        </div>
      `
      cards.appendChild(c)
    })
  }
  load()
  setInterval(load, 5000)  // Auto-refresh every 5 seconds
})
