/**
 * Malaria Detection — Frontend JavaScript
 * Handles: drag & drop, image preview, form loading state
 */

document.addEventListener('DOMContentLoaded', () => {

    // ── Element refs ────────────────────────────────────────────────────────
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewCont = document.getElementById('previewContainer');
    const previewImg = document.getElementById('previewImage');
    const previewName = document.getElementById('previewFilename');
    const btnRemove = document.getElementById('btnRemove');
    const analyzeForm = document.getElementById('analyzeForm');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('spinner');

    if (!dropZone) return;    // Guard: only run on index page

    // ── Drag & Drop ─────────────────────────────────────────────────────────
    ['dragenter', 'dragover'].forEach(ev => {
        dropZone.addEventListener(ev, e => {
            e.preventDefault(); e.stopPropagation();
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(ev => {
        dropZone.addEventListener(ev, e => {
            e.preventDefault(); e.stopPropagation();
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    // ── File Input Change ───────────────────────────────────────────────────
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    // ── Handle File ─────────────────────────────────────────────────────────
    function handleFile(file) {
        const allowedTypes = ['image/png', 'image/jpeg', 'image/bmp', 'image/tiff', 'image/gif'];
        if (!allowedTypes.includes(file.type) && !file.name.match(/\.(png|jpg|jpeg|bmp|tiff|tif)$/i)) {
            showAlert('Unsupported file type. Please upload PNG, JPG, or BMP.', 'error');
            return;
        }

        // Assign to input (for form submission)
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;

        // Show preview
        const reader = new FileReader();
        reader.onload = e => {
            previewImg.src = e.target.result;
            previewName.textContent = file.name + ' (' + formatBytes(file.size) + ')';
            previewCont.classList.add('visible');
            btnAnalyze.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // ── Remove Preview ──────────────────────────────────────────────────────
    if (btnRemove) {
        btnRemove.addEventListener('click', () => {
            fileInput.value = '';
            previewImg.src = '';
            previewCont.classList.remove('visible');
            btnAnalyze.disabled = true;
        });
    }

    // ── Form Submit — Loading State ─────────────────────────────────────────
    if (analyzeForm) {
        analyzeForm.addEventListener('submit', e => {
            if (!fileInput.files.length) {
                e.preventDefault();
                showAlert('Please select an image first.', 'error');
                return;
            }
            btnText.textContent = 'Analysing…';
            spinner.style.display = 'block';
            btnAnalyze.disabled = true;
        });
    }

    // ── Confidence Bar Animation (result page) ──────────────────────────────
    const fillBar = document.getElementById('confidenceBarFill');
    if (fillBar) {
        const target = parseFloat(fillBar.dataset.target || '0');
        setTimeout(() => { fillBar.style.width = target + '%'; }, 200);
    }

    // ── Helpers ─────────────────────────────────────────────────────────────
    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    }

    function showAlert(msg, type = 'error') {
        const container = document.querySelector('.flash-container')
            || document.body;
        const div = document.createElement('div');
        div.className = `flash ${type}`;
        div.innerHTML = `<span>${type === 'error' ? '⚠️' : '✅'}</span> ${msg}`;
        container.prepend(div);
        setTimeout(() => div.remove(), 5000);
    }

});
