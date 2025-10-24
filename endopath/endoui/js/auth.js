// Lightweight frontend auth helpers.
// Requires backend endpoints: POST /auth/login, POST /auth/logout, GET /auth/session

async function login(username, password){
  return API.post('/auth/login', { username, password })
}

async function logout(){
  // best-effort - backend should clear cookie
  try{
    await API.post('/auth/logout', {})
  }catch(e){
    // ignore
  }
  // redirect to login
  location.href = 'login.html'
}

async function getSession(){
  try{
    const s = await API.get('/auth/session')
    return s
  }catch(err){
    return null
  }
}

// Auto-run on pages: if there's a login form, wire it; otherwise ensure session
document.addEventListener('DOMContentLoaded', ()=>{
  const form = document.getElementById('loginForm')
  if (form){
    form.addEventListener('submit', async (e)=>{
      e.preventDefault()
      const fd = new FormData(form)
      const username = fd.get('username')
      const password = fd.get('password')
      try{
        await login(username, password)
        showMessage('Login successful — redirecting...', 'success')
        setTimeout(()=> location.href = 'index.html', 700)
      }catch(err){
        console.error(err)
        showMessage('Login failed: '+err.message, 'error')
      }
    })
    return
  }

  // Not a login page: ensure session. If no session, redirect to login.
  (async ()=>{
    const s = await getSession()
    if (!s){
      if (!location.pathname.endsWith('login.html')){
        location.href = 'login.html'
        return
      }
    }
  })()
})

// expose helpers
window.Auth = { login, logout, getSession }


// expose helpers
window.Auth = { login, logout, getSession }
