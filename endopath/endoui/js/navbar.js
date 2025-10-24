// Shared navigation component
function renderNavbar(currentPage) {
  const navHTML = `
    <nav class="navbar">
      <div class="navbar-content">
        <a href="index.html" class="navbar-brand">🔬 EndoAI</a>
        <div class="navbar-user" id="navUser">
          <span id="navUsername"></span>
          <a href="#" onclick="logout(); return false;" style="color: white; margin-left: 1rem;">Logout</a>
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
