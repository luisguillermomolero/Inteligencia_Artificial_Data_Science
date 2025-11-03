const API_BASE = 'http://127.0.0.1:8000';
const fileInput = document.getElementById('fileInput');
const sendBtn = document.getElementById('sendBtn');
const preview = document.getElementById('preview');
const result = document.getElementById('result');
let previewImg = null;

// Etiquetas COCO (alineadas a los IDs de torchvision; índice 0 vacío)
const COCO_LABELS = [
  '', 'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
  'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
  'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
  'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
  'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
  'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
  'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
  'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
];

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  const img = document.createElement('img');
  img.src = URL.createObjectURL(file);
  preview.innerHTML = '';
  preview.appendChild(img);
  previewImg = img;
  result.innerText = '';
  sendBtn.disabled = false;
});

sendBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) { alert('Selecciona una imagen'); return; }
  const fd = new FormData();
  fd.append('file', file);

  try {
    const resp = await fetch(`${API_BASE}/predict/upload`, { method: 'POST', body: fd });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Error ${resp.status}: ${text}`);
    }
    const json = await resp.json();
    const name = json.class_name || `id:${json.class_id}`;
    result.innerText = `Predicción: ${name} — ${(json.score*100).toFixed(1)}%`;
  } catch (e) {
    result.innerText = `Fallo en clasificación: ${e}`;
  }
});

