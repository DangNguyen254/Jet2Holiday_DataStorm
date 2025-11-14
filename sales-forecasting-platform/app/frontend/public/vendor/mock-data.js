(function(){
  const LS_KEY = 'mockDataset.v1';
  const regionMap = { 'Hà Nội':'North', 'Đà Nẵng':'Central', 'Hồ Chí Minh':'South' };

  const seedProducts = [
    { sku: 'SKU-A', name: 'Áo thun', category: 'Gia dụng', region: 'South', store: 'HCM-01' },
    { sku: 'SKU-B', name: 'Cà phê lon', category: 'Đồ uống', region: 'South', store: 'HCM-01' },
    { sku: 'SKU-C', name: 'Mì ly', category: 'Thực phẩm', region: 'North', store: 'HN-02' },
    { sku: 'SKU-D', name: 'Khăn giấy', category: 'Gia dụng', region: 'Central', store: 'DN-03' },
    { sku: 'SKU-E', name: 'Nước suối', category: 'Đồ uống', region: 'Central', store: 'DN-03' },
    { sku: 'SKU-F', name: 'Bánh quy', category: 'Thực phẩm', region: 'South', store: 'HCM-01' },
    { sku: 'SKU-G', name: 'Sữa chua', category: 'Thực phẩm', region: 'North', store: 'HN-02' },
    { sku: 'SKU-H', name: 'Nước ngọt', category: 'Đồ uống', region: 'Central', store: 'DN-03' },
    { sku: 'SKU-I', name: 'Bột giặt', category: 'Gia dụng', region: 'South', store: 'HCM-01' },
    { sku: 'SKU-J', name: 'Dầu ăn', category: 'Thực phẩm', region: 'North', store: 'HN-02' },
    { sku: 'SKU-K', name: 'Trà xanh', category: 'Đồ uống', region: 'South', store: 'HCM-01' },
    { sku: 'SKU-L', name: 'Bánh mì', category: 'Thực phẩm', region: 'Central', store: 'DN-03' },
    { sku: 'SKU-M', name: 'Bia lon', category: 'Đồ uống', region: 'North', store: 'HN-02' },
    { sku: 'SKU-N', name: 'Sữa tươi', category: 'Thực phẩm', region: 'Central', store: 'DN-03' },
    { sku: 'SKU-O', name: 'Kem đánh răng', category: 'Gia dụng', region: 'South', store: 'HCM-01' },
    { sku: 'SKU-P', name: 'Nước giặt', category: 'Gia dụng', region: 'North', store: 'HN-02' },
    { sku: 'SKU-Q', name: 'Snack khoai tây', category: 'Thực phẩm', region: 'Central', store: 'DN-03' },
    { sku: 'SKU-R', name: 'Nước tăng lực', category: 'Đồ uống', region: 'South', store: 'HCM-01' },
    { sku: 'SKU-S', name: 'Mì gói', category: 'Thực phẩm', region: 'North', store: 'HN-02' },
    { sku: 'SKU-T', name: 'Bánh cracker', category: 'Thực phẩm', region: 'Central', store: 'DN-03' },
    { sku: 'SKU-U', name: 'Bánh bông lan', category: 'Thực phẩm', region: 'South', store: 'HCM-02' },
    { sku: 'SKU-V', name: 'Nước ép', category: 'Đồ uống', region: 'South', store: 'HCM-03' },
    { sku: 'SKU-W', name: 'Gạo thơm', category: 'Thực phẩm', region: 'North', store: 'HN-01' },
    { sku: 'SKU-X', name: 'Trà hoa quả', category: 'Đồ uống', region: 'North', store: 'HN-03' },
    { sku: 'SKU-Y', name: 'Giấy vệ sinh', category: 'Gia dụng', region: 'Central', store: 'DN-01' },
    { sku: 'SKU-Z', name: 'Nước lau sàn', category: 'Gia dụng', region: 'Central', store: 'DN-02' }
  ];

  function daysAgo(n){ const d=new Date(); d.setDate(d.getDate()-n); d.setHours(0,0,0,0); return d; }
  function toISODate(d){ return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`; }

  function generateDefault(days=60){
    const items=[];
    for (let d=0; d<days; d++){
      const date = daysAgo(d);
      const isWeekend = [0,6].includes(date.getDay());
      const temp = 24 + (Math.sin((d/7)*Math.PI)*6) + (isWeekend?1.5:0);
      const rain = Math.max(0, 10 + (Math.cos((d/5)*Math.PI)*12) + (Math.random()*8-4));
      for (const p of seedProducts){
        const base = 18 + (p.category==='Đồ uống'?12:0) + (isWeekend?5:0);
        const weatherFactor = 1 + (p.category==='Đồ uống' ? (temp-24)*0.012 : 0) - (rain>15?0.08:0) - (rain>25?0.07:0);
        const forecast = Math.max(1, Math.round((base + Math.random()*6-3) * weatherFactor));
        const actual = Math.max(0, Math.round(forecast + (Math.random()*12-6)));
        const stock = Math.max(0, 60 - (actual + d*2) + Math.floor(Math.random()*8-4));
        items.push({
          date: toISODate(date),
          region: p.region, store: p.store, category: p.category,
          sku: p.sku, product: p.name,
          forecast, actual,
          stockRemaining: stock,
          weather: { rain: +rain.toFixed(1), temp: +temp.toFixed(1) }
        });
      }
    }
    return items;
  }

  function revive(items){
    return (items||[]).map(r => ({
      ...r,
      date: new Date(r.date + 'T00:00:00')
    }));
  }

  function serialize(items){
    return (items||[]).map(r => ({
      date: (r.date instanceof Date) ? toISODate(r.date) : String(r.date).slice(0,10),
      region: r.region, store: r.store, category: r.category,
      sku: r.sku, product: r.product,
      forecast: +r.forecast||0, actual: +r.actual||0,
      stockRemaining: +r.stockRemaining||0,
      weather: r.weather && typeof r.weather === 'object' ? { rain: +r.weather.rain||0, temp: +r.weather.temp||0 } : undefined
    }));
  }

  function load(){
    try {
      const txt = localStorage.getItem(LS_KEY);
      if (!txt) return null;
      const arr = JSON.parse(txt);
      if (!Array.isArray(arr)) return null;
      return revive(arr);
    } catch(e){ return null; }
  }

  function save(items){
    try { localStorage.setItem(LS_KEY, JSON.stringify(serialize(items))); } catch(e) {}
  }

  function getProductsFrom(items){
    const map = {};
    for (const r of items){ if (!map[r.sku]) map[r.sku] = { sku: r.sku, name: r.product, category: r.category, region: r.region, store: r.store }; }
    return Object.values(map);
  }

  function getDataset(){
    let data = load();
    if (!data){
      const def = generateDefault(60);
      save(def);
      data = revive(def);
    }
    return data;
  }

  function setDataset(items){
    // Expect date as Date or ISO string
    save(items);
  }

  function regenerate(days){
    const def = generateDefault(days||60);
    save(def);
    return revive(def);
  }

  window.MockData = {
    getDataset,
    setDataset,
    regenerate,
    getProducts: () => getProductsFrom(getDataset()),
    regionMap,
    LS_KEY
  };
})();
