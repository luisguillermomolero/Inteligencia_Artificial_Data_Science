const fileInput = document.getElementById('fileInput');
const sendBtn = document.getElementById('sendBtn');
const preview = document.getElementById('preview');
const result = document.getElementById('result');

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  const img = document.createElement('img');
  img.src = URL.createObjectURL(file);
  preview.innerHTML = '';
  preview.appendChild(img);
});

sendBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) { alert('Selecciona una imagen'); return; }
  const fd = new FormData();
  fd.append('file', file);

  const resp = await fetch('http://localhost:8000/predict/upload', {
    method: 'POST', body: fd
  });
  const json = await resp.json();
  result.innerText = JSON.stringify(json, null, 2);
});
