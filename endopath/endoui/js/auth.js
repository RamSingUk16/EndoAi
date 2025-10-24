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

function renderNav(user){
  let nav = document.getElementById('nav')
  if (!nav){
    nav = document.createElement('nav')
    nav.id = 'nav'
    document.body.insertBefore(nav, document.body.firstChild)
  }
  nav.innerHTML = ''
  const home = document.createElement('a')
  home.href = 'upload.html'
  home.textContent = 'Upload'
  nav.appendChild(home)
  const sep = document.createTextNode(' | ')
  nav.appendChild(sep)
  const results = document.createElement('a')
  results.href = 'results.html'
  results.textContent = 'Results'
  nav.appendChild(results)
  if (user && user.username){
    const right = document.createElement('span')
    right.style.float = 'right'
    right.innerHTML = `Signed in as <strong>${user.username}</strong> <button id="logoutBtn">Logout</button>`
    nav.appendChild(right)
    document.getElementById('logoutBtn').addEventListener('click', logout)
  } else {
    const loginLink = document.createElement('a')
    loginLink.href = 'login.html'
    loginLink.textContent = 'Login'
    nav.appendChild(loginLink)
  }
}

// Auto-run on pages: if there's a login form, wire it; otherwise attempt to get session and render nav
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
        setTimeout(()=> location.href = 'upload.html', 700)
      }catch(err){
        console.error(err)
        showMessage('Login failed: '+err.message, 'error')
      }
    })
    return
  }

  // Not a login page: ensure session and render nav. If no session, redirect to login.
  (async ()=>{
    const s = await getSession()
    if (!s){
      // allow anonymous access to results page for now? follow plan: redirect to login
      if (!location.pathname.endsWith('login.html')){
        location.href = 'login.html'
        return
      }
    }
    renderNav(s || {})
  })()
})

// expose helpers
window.Auth = { login, logout, getSession }
