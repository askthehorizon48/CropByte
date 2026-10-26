# CropByte

CropByte is a multilingual crop-image classifier with a local Flask API. It uses
`cropbyte_model.tflite` and the matching class names in `cropbyte_labels.json`.

## Run on Windows

Use Python 3.11 for the most reliable TensorFlow package compatibility:

```powershell
cd "C:\Users\india\OneDrive\Documents\crop disease predictiom\CropByte"
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000/>. Keep the terminal open while using the site;
press **Ctrl+C** there to stop the server. Do not launch the pages with
`file://` or `python -m http.server` because image analysis requires the Flask
API.

## Retrain the classifier

The model currently in the project was trained with a different preprocessing
pipeline and repeatedly predicts `Corn___healthy`. The app refuses to serve that
stale model so it cannot present those predictions as diagnoses. The training
dataset is not included in this project. The official
[PlantVillage dataset](https://huggingface.co/datasets/mohanty/PlantVillage)
provides the source images and grouped train/test splits. Its source archive is
about 2.03 GiB; make sure there is enough free disk space before downloading.
The downloader below keeps the 15 classes this project supports and maps them
to the matching class names.

Run these commands from the CropByte folder:

```powershell
cd "C:\Users\india\OneDrive\Documents\crop disease predictiom\CropByte"
py download_dataset.py
.\.venv\Scripts\python.exe train.py --data-dir ".\data\PlantVillage" --epochs 10
```

The downloader uses the source's predefined color train/test split to avoid
putting images of the same leaf in both sets. It downloads the full archive,
extracts only the needed classes, and stores them in `data\PlantVillage`.

Training applies MobileNetV2's `[-1, 1]` preprocessing to both the training
images and web uploads, and writes the model, labels, and preprocessing metadata
together. Restart the web app after training. Use representative validation
images to check accuracy before relying on predictions.

If TensorFlow cannot be installed on your Windows version, run the app in
Windows Subsystem for Linux (WSL 2) with Python 3.11 and the same requirements.

## Features

- Language selection in English, Hindi, Malayalam, Tamil, Telugu, or Kannada.
- Crop photo upload or camera capture, preview, and image-based classification.
- Confidence and alternative class scores, spoken advice, and a downloadable
  text report.
- Image uploads are processed in memory and limited to 10 MB.

The model supports the 15 classes listed in `cropbyte_labels.json`. `labels.json`
belongs to a different 38-class model and is not used for these predictions. AI
predictions are estimates, not confirmed diagnoses; check symptoms with a local
agricultural expert before choosing a treatment. Check the dataset host's terms
and cite the PlantVillage dataset paper if you reuse its images or resulting model.
