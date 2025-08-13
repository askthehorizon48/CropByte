function setLanguage(lang) {
  localStorage.setItem('selectedLanguage', lang);
  window.location.href = "main.html";
}

function applyTranslations(lang) {
  const t = translations[lang] || translations['en'];
  if (document.getElementById('title')) {
    document.getElementById('title').textContent = t.title;
  }
  if (document.getElementById('description')) {
    document.getElementById('description').textContent = t.description;
  }
}

// On page load, apply the saved language
document.addEventListener('DOMContentLoaded', () => {
  const lang = localStorage.getItem('selectedLanguage') || 'en';
  if (typeof translations !== "undefined") {
    // For main.html
    if (document.getElementById('heroTitle')) {
      document.getElementById('heroTitle').innerHTML = translations[lang].heroTitle;
    }
    if (document.getElementById('heroDesc')) {
      document.getElementById('heroDesc').innerHTML = translations[lang].heroDesc;
    }
    if (document.getElementById('uploadLabel')) {
      document.getElementById('uploadLabel').textContent = translations[lang].uploadLabel;
    }
    if (document.getElementById('notesLabel')) {
      document.getElementById('notesLabel').textContent = translations[lang].notesLabel;
    }
    if (document.getElementById('notes')) {
      document.getElementById('notes').placeholder = translations[lang].notesPlaceholder;
    }
    if (document.getElementById('analyzeBtn')) {
      document.getElementById('analyzeBtn').textContent = translations[lang].analyzeBtn;
    }
    if (document.getElementById('detectedDiseaseLabel')) {
      document.getElementById('detectedDiseaseLabel').childNodes[0].nodeValue = translations[lang].detectedDisease + " ";
    }
    if (document.getElementById('adviceLabel')) {
      document.getElementById('adviceLabel').childNodes[0].nodeValue = translations[lang].advice + " ";
    }
    if (document.getElementById('hearAdviceBtn')) {
      document.getElementById('hearAdviceBtn').textContent = translations[lang].hearAdvice;
    }
    if (document.getElementById('downloadReportBtn')) {
      document.getElementById('downloadReportBtn').textContent = translations[lang].downloadReport;
    }
  }
  const takePhotoBtn = document.getElementById('takePhotoBtn');
  const imageInput = document.getElementById('imageInput');
  if (takePhotoBtn && imageInput) {
    takePhotoBtn.addEventListener('click', function() {
      imageInput.click();
    });
  }
});

window.onload = function() {
  // Image preview
  const imageInput = document.getElementById('imageInput');
  if (imageInput) {
    imageInput.addEventListener('change', function(e) {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function(evt) {
          document.getElementById('imagePreview').innerHTML =
            `<img src="${evt.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(file);
      }
    });
  }
};
