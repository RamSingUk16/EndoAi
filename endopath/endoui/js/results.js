document.addEventListener('DOMContentLoaded', ()=>{
  const cards = document.getElementById('cards')
  const summary = document.getElementById('summary')
  async function load(){
    try{
      const data = await API.get('/cases')
      render(data)
    }catch(err){
      console.error(err)
      showMessage('Failed to load results: '+err.message,'error')
    }
  }
  function render(list){
    if (!Array.isArray(list)) return
    summary.textContent = `Total: ${list.length}`
    cards.innerHTML = ''
    list.forEach(item=>{
      const c = document.createElement('div')
      c.className = 'card'
      c.innerHTML = `<strong>Slide #${item.id}</strong><div>Status: ${item.status||'n/a'}</div>`
      cards.appendChild(c)
    })
  }
  load()
  setInterval(load, 5000)
})
