// Shared navigation component
function renderNavbar(currentPage) {
  const navHTML = `
    <nav class="navbar">
      <div class="navbar-content">
        <a href="index.html" class="navbar-brand">🔬 EndoAI</a>
        <div class="navbar-right">
          <ul class="navbar-nav">
            <li><a href="about.html" class="${currentPage === 'about' ? 'active' : ''}">About</a></li>
            <li><a href="technical.html" class="${currentPage === 'technical' ? 'active' : ''}">Technical</a></li>
          </ul>
          <div class="navbar-user" id="navUser">
            <span id="navUsername"></span>
            <a href="#" onclick="logout(); return false;" class="logout-link">Logout</a>
          </div>
        </div>
      </div>
    </nav>
  `;
  
  // Insert navbar at the beginning of body
  document.body.insertAdjacentHTML('afterbegin', navHTML);
  
  // Load and display current user
  loadCurrentUser();
}

async function loadCurrentUser() {
  try {
    const user = await getSession();
    if (user) {
      const usernameEl = document.getElementById('navUsername');
      if (usernameEl) {
        usernameEl.textContent = `👤 ${user.username}${user.is_admin ? ' (Admin)' : ''}`;
      }
    }
  } catch (err) {
    console.log('Not logged in');
  }
}
