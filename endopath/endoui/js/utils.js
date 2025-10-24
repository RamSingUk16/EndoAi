function showMessage(msg, type='info'){
  const el = document.getElementById('msg') || document.getElementById('result') || document.body
  const p = document.createElement('div')
  p.textContent = msg
  p.className = 'toast '+type
  el.appendChild(p)
  setTimeout(()=>p.remove(),5000)
}

window.showMessage = showMessage
