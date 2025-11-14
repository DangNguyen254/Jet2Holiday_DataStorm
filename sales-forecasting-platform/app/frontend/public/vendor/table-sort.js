// Generic table sorting utility
// Usage: add class 'sortable' to <table>. Each <th> becomes clickable to sort.
// Auto-detect column type: number, date (ISO or dd/mm/yyyy), and text.
// Keeps stable sort by pairing value + original index.
(function(){
  function parseDate(str){
    if(!str) return NaN;
    str = str.trim();
    // ISO
    const iso = Date.parse(str);
    if(!isNaN(iso)) return iso;
    // dd/mm/yyyy
    const m = str.match(/^([0-3]?\d)[\/\-]([0-1]?\d)[\/\-](\d{4})$/);
    if(m){
      const d = new Date(+m[3], +m[2]-1, +m[1]);
      return d.getTime();
    }
    return NaN;
  }
  function detectType(values){
    let numCount=0, dateCount=0;
    values.forEach(v=>{
      const n = parseFloat(v.replace(/,/g,''));
      if(!isNaN(n) && v.match(/^[\s\-+]?\d*[\.,]?\d+%?$/)) numCount++;
      const dt = parseDate(v);
      if(!isNaN(dt)) dateCount++;
    });
    if(dateCount/values.length > 0.6) return 'date';
    if(numCount/values.length > 0.6) return 'number';
    return 'text';
  }
  function getCellValue(td){ return td ? td.textContent.trim() : ''; }
  function sortTable(table, colIndex, dir){
    const tbody = table.tBodies[0]; if(!tbody) return;
    const rows = Array.from(tbody.rows);
    const sampleVals = rows.map(r => getCellValue(r.cells[colIndex]));
    const type = detectType(sampleVals);
    const mapped = rows.map((r,i)=>{
      const raw = getCellValue(r.cells[colIndex]);
      let val = raw.toLowerCase();
      if(type==='number'){
        const num = parseFloat(raw.replace(/[%,$]/g,'').replace(/,/g,'.'));
        val = isNaN(num)? -Infinity : num;
      } else if(type==='date'){
        const dt = parseDate(raw);
        val = isNaN(dt)? -Infinity : dt;
      }
      return {r, i, val};
    });
    mapped.sort((a,b)=>{
      if(a.val < b.val) return dir==='asc'? -1:1;
      if(a.val > b.val) return dir==='asc'? 1:-1;
      return a.i - b.i; // stable
    });
    mapped.forEach(m => tbody.appendChild(m.r));
  }
  function clearIndicators(table){
    table.querySelectorAll('th').forEach(th=> th.classList.remove('sort-asc','sort-desc'));
  }
  function enhance(table){
    if(table.dataset.sortableEnhanced) return; // idempotent
    table.dataset.sortableEnhanced = '1';
    const thead = table.tHead; if(!thead) return;
    thead.querySelectorAll('th').forEach((th, idx)=>{
      th.style.cursor='pointer';
      th.addEventListener('click', ()=>{
        const currentDir = th.classList.contains('sort-asc') ? 'asc' : (th.classList.contains('sort-desc') ? 'desc' : null);
        const newDir = currentDir==='asc' ? 'desc' : 'asc';
        clearIndicators(table);
        th.classList.add(newDir==='asc'?'sort-asc':'sort-desc');
        sortTable(table, idx, newDir);
      });
    });
  }
  function enhanceAll(){
    document.querySelectorAll('table.sortable').forEach(enhance);
  }
  // Mutation observer to auto-enhance freshly inserted tables
  const observer = new MutationObserver(()=> enhanceAll());
  observer.observe(document.documentElement, {childList:true, subtree:true});
  document.addEventListener('DOMContentLoaded', enhanceAll);
  window.TableSortEnhance = enhanceAll; // manual trigger after dynamic rebuild
})();

// Minimal dark mode indicators (add via CSS if desired)
// th.sort-asc::after { content: ' \25B2'; }
// th.sort-desc::after { content: ' \25BC'; }
