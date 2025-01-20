document.addEventListener('DOMContentLoaded', function() {
    const links = document.querySelectorAll('.nav-link');
    links.forEach(link => {
        link.addEventListener('click', function() {
            links.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // Set the active link based on the current URL
    const currentPath = window.location.pathname;
    if (currentPath === '/') {
        document.getElementById('home-link').classList.add('active');
    } else if (currentPath === '/upload-form/') {
        document.getElementById('upload-link').classList.add('active');
    }
});

function displayValidationErrors(errors) {
    const validationSummary = document.getElementById('validation-summary');
    const validationErrors = document.getElementById('validation-errors');
    validationErrors.innerHTML = '';
    errors.forEach(error => {
        const li = document.createElement('li');
        li.textContent = error;
        validationErrors.appendChild(li);
    });
    validationSummary.style.display = 'block';
}

function validateFileUpload(event) {
    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];
    const validTypes = ['image/png', 'image/jpeg'];
    const maxSize = 5 * 1024 * 1024; // 5 MB
    const errors = [];

    if (!file) {
        errors.push('Please select a file.');
    } else {
        if (!validTypes.includes(file.type)) {
            errors.push('Invalid file type. Only .png and .jpeg files are allowed.');
        }
        if (file.size > maxSize) {
            errors.push('File size exceeds 5 MB.');
        }
    }

    if (errors.length > 0) {
        displayValidationErrors(errors);
        fileInput.classList.add('is-invalid');
        event.preventDefault();
    } else {
        fileInput.classList.remove('is-invalid');
        document.getElementById('validation-summary').style.display = 'none';
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', validateFileUpload);
    }

    const links = document.querySelectorAll('.nav-link');
    links.forEach(link => {
        link.addEventListener('click', function() {
            links.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });

    const currentPath = window.location.pathname;
    if (currentPath === '/') {
        document.getElementById('home-link').classList.add('active');
    } else if (currentPath === '/upload-form/') {
        document.getElementById('upload-link').classList.add('active');
    }
});
document.addEventListener('htmx:afterRequest', function(event) {
    if (event.detail.target.id === 'result' && event.detail.xhr.status === 400) {
        const response = JSON.parse(event.detail.xhr.responseText);
        if (response.errors) {
            displayValidationErrors(response.errors);
        }
    }
});

