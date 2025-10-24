document.addEventListener('DOMContentLoaded', ()=>{
  const form = document.getElementById('loginForm')
  if (!form) return
  form.addEventListener('submit', async (e)=>{
    e.preventDefault()
    const fd = new FormData(form)
    const username = fd.get('username')
    const password = fd.get('password')
    try{
      await API.post('/auth/login', { username, password })
      showMessage('Login successful — redirecting...', 'success')
      setTimeout(()=> location.href = 'upload.html', 700)
    }catch(err){
      console.error(err)
      showMessage('Login failed: '+err.message, 'error')
    }
  })
})
