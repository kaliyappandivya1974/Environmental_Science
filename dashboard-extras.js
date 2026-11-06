const globalSearch=document.querySelector('#global-search');
const navLinks=[...document.querySelectorAll('.nav-item')];
navLinks.forEach(link=>link.title=link.textContent.trim());
globalSearch.addEventListener('input',()=>{
  const query=globalSearch.value.trim().toLocaleLowerCase();
  navLinks.forEach(link=>link.classList.toggle('filtered-out',query!==''&&!link.textContent.toLocaleLowerCase().includes(query)));
});
globalSearch.addEventListener('keydown',event=>{
  if(event.key!=='Enter')return;
  const match=navLinks.find(link=>!link.classList.contains('filtered-out'));
  if(match){match.click();globalSearch.value='';navLinks.forEach(link=>link.classList.remove('filtered-out'));}
});
