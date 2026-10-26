function setLanguage(lang) {
  localStorage.setItem('selectedLanguage', lang);
  window.location.href = "main.html";
}

document.addEventListener('DOMContentLoaded', () => {
  if (typeof translations === "undefined") return;
  const lang = localStorage.getItem('selectedLanguage') || 'en';
  const t = translations[lang] || translations.en;
  document.documentElement.lang = lang;

  if (document.getElementById('heroTitle')) {
    document.getElementById('heroTitle').textContent = t.heroTitle;
    document.getElementById('heroDesc').textContent = t.heroDesc;
    document.getElementById('uploadLabel').textContent = t.uploadLabel;
    document.getElementById('notesLabel').textContent = t.notesLabel;
    document.getElementById('notes').placeholder = t.notesPlaceholder;
    document.getElementById('analyzeBtn').textContent = t.analyzeBtn;
    document.getElementById('detectedDiseaseLabel').textContent = t.detectedDisease;
    document.getElementById('confidenceLabel').textContent = t.confidence;
    document.getElementById('adviceLabel').textContent = t.advice;
    document.getElementById('hearAdviceBtn').textContent = t.hearAdvice;
    document.getElementById('downloadReportBtn').textContent = t.downloadReport;
    document.getElementById('disclaimer').textContent = t.disclaimer;
    document.getElementById('takePhotoBtn').lastChild.textContent = ` ${t.takePhoto}`;
    document.getElementById('resultHeading').textContent = t.resultHeading;
    document.getElementById('alternativesLabel').textContent = t.alternatives;
    document.getElementById('changeLanguageBtn').textContent = t.changeLanguage;
    document.getElementById('modelSetupTitle').textContent = t.modelSetupTitle;
    document.getElementById('modelSetupMessage').textContent = t.modelSetupMessage;
    document.getElementById('modelSetupDataset').textContent = t.modelSetupDataset;
    document.getElementById('modelSetupTrain').textContent = t.modelSetupTrain;
    document.getElementById('modelSetupSpace').textContent = t.modelSetupSpace;
    document.getElementById('readmeLink').textContent = t.readmeLink;
    document.getElementById('datasetLink').textContent = t.datasetLink;
    document.getElementById('retryModelBtn').textContent = t.retryModel;
    document.getElementById('technicalDetailsLabel').textContent = t.technicalDetails;
  }

  const imageInput = document.getElementById('imageInput');
  const imagePreview = document.getElementById('imagePreview');
  const form = document.getElementById('cropForm');
  const analyzeButton = document.getElementById('analyzeBtn');
  const statusMessage = document.getElementById('statusMessage');
  const modelSetup = document.getElementById('modelSetup');
  const retryModelButton = document.getElementById('retryModelBtn');
  const resultSection = document.getElementById('result');
  let currentResult = null;
  let previewUrl = null;
  let modelReady = false;
  let analyzing = false;

  analyzeButton.disabled = true;
  const updateAnalyzeButton = () => {
    analyzeButton.disabled = !modelReady || !imageInput.files[0] || analyzing;
  };

  document.getElementById('takePhotoBtn').addEventListener('click', () => imageInput.click());
  document.getElementById('changeLanguageBtn').addEventListener('click', () => {
    window.location.href = 'index.html';
  });
  retryModelButton.addEventListener('click', checkModel);

  imageInput.addEventListener('change', () => {
    const file = imageInput.files[0];
    resultSection.hidden = true;
    currentResult = null;
    statusMessage.textContent = '';
    updateAnalyzeButton();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    imagePreview.replaceChildren();
    imagePreview.hidden = !file;
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      imageInput.value = '';
      imagePreview.hidden = true;
      statusMessage.textContent = t.invalidImage;
      updateAnalyzeButton();
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      imageInput.value = '';
      imagePreview.hidden = true;
      statusMessage.textContent = t.imageTooLarge;
      updateAnalyzeButton();
      return;
    }
    previewUrl = URL.createObjectURL(file);
    const image = document.createElement('img');
    image.src = previewUrl;
    image.alt = t.imagePreview;
    imagePreview.append(image);
    updateAnalyzeButton();
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const file = imageInput.files[0];
    if (!modelReady) {
      statusMessage.textContent = t.modelUnavailable;
      modelSetup.hidden = false;
      return;
    }
    if (!file) {
      statusMessage.textContent = t.chooseImage;
      imageInput.focus();
      return;
    }

    analyzing = true;
    updateAnalyzeButton();
    analyzeButton.setAttribute('aria-busy', 'true');
    statusMessage.textContent = t.analyzing;
    resultSection.hidden = true;
    try {
      const body = new FormData();
      body.append('image', file);
      const response = await fetch('/api/predict', { method: 'POST', body });
      const result = await response.json();
      if (!response.ok) {
        if (response.status === 503) {
          modelReady = false;
          modelSetup.hidden = false;
          document.getElementById('modelDiagnostic').textContent = result.error || '';
          throw new Error(t.modelUnavailable);
        }
        throw new Error(result.error || t.analysisFailed);
      }

      currentResult = result;
      document.getElementById('diseaseName').textContent = result.prediction;
      document.getElementById('confidenceValue').textContent = `${(result.confidence * 100).toFixed(1)}%`;
      document.getElementById('confidenceNotice').textContent =
        result.confidence < 0.6 ? t.lowConfidence : '';
      document.getElementById('confidenceNotice').hidden = result.confidence >= 0.6;
      document.getElementById('adviceText').textContent = result.healthy ? t.healthyAdvice : t.diseaseAdvice;
      document.getElementById('alternatives').hidden = result.alternatives.length === 0;
      const alternativesList = document.getElementById('alternativesList');
      alternativesList.replaceChildren();
      result.alternatives.forEach((alternative) => {
        const item = document.createElement('li');
        item.textContent = `${alternative.prediction} (${(alternative.confidence * 100).toFixed(1)}%)`;
        alternativesList.append(item);
      });
      resultSection.hidden = false;
      statusMessage.textContent = '';
    } catch (error) {
      statusMessage.textContent = error.message || t.analysisFailed;
    } finally {
      analyzing = false;
      updateAnalyzeButton();
      analyzeButton.removeAttribute('aria-busy');
    }
  });

  document.getElementById('hearAdviceBtn').addEventListener('click', () => {
    if (!currentResult || !('speechSynthesis' in window)) {
      statusMessage.textContent = t.speechUnavailable;
      return;
    }
    window.speechSynthesis.cancel();
    const speech = new SpeechSynthesisUtterance(
      `${currentResult.prediction}. ${currentResult.healthy ? t.healthyAdvice : t.diseaseAdvice}`
    );
    speech.lang = t.speechLocale;
    window.speechSynthesis.speak(speech);
  });

  document.getElementById('downloadReportBtn').addEventListener('click', () => {
    if (!currentResult) return;
    const report = [
      'CropByte - Plant health report',
      `Date: ${new Date().toLocaleString()}`,
      `Result: ${currentResult.prediction}`,
      `Confidence: ${(currentResult.confidence * 100).toFixed(1)}%`,
      `Advice: ${currentResult.healthy ? t.healthyAdvice : t.diseaseAdvice}`,
      `Notes: ${document.getElementById('notes').value.trim() || '-'}`,
      t.disclaimer,
    ].join('\n\n');
    const url = URL.createObjectURL(new Blob([report], { type: 'text/plain;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'cropbyte-report.txt';
    link.hidden = true;
    document.body.append(link);
    link.click();
    window.setTimeout(() => {
      URL.revokeObjectURL(url);
      link.remove();
    }, 1000);
  });

  async function checkModel() {
    retryModelButton.disabled = true;
    statusMessage.textContent = t.checkingModel;
    try {
      const response = await fetch('/api/health');
      const health = await response.json();
      if (!response.ok || !health.ready) {
        document.getElementById('modelDiagnostic').textContent = health.error || '';
        throw new Error(t.modelUnavailable);
      }
      modelReady = true;
      modelSetup.hidden = true;
      document.getElementById('modelDiagnostic').textContent = '';
      statusMessage.textContent = t.modelReady;
    } catch (error) {
      modelReady = false;
      modelSetup.hidden = false;
      statusMessage.textContent = error.message || t.modelUnavailable;
    } finally {
      retryModelButton.disabled = false;
      updateAnalyzeButton();
    }
  }

  checkModel();
});
