import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "PlantVillage"
ARCHIVE_URL = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip?download=true"
SPLIT_URL = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_{split}.txt?download=true"
SOURCE_CLASSES = {
    "Apple___Apple_scab": "Apple___Apple_scab",
    "Apple___Black_rot": "Apple___Black_rot",
    "Apple___Cedar_apple_rust": "Apple___Cedar_apple_rust",
    "Apple___healthy": "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_": "Corn___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn___Northern_Leaf_Blight",
    "Corn_(maize)___healthy": "Corn___healthy",
    "Grape___Black_rot": "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)": "Grape___Esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape___Leaf_blight",
    "Grape___healthy": "Grape___healthy",
    "Tomato___Early_blight": "Tomato___Early_blight",
    "Tomato___Late_blight": "Tomato___Late_blight",
    "Tomato___healthy": "Tomato___healthy",
}
CHUNK_SIZE = 4 * 1024 * 1024


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "CropByte dataset setup"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        last_report = started
        with destination.open("wb") as output:
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5:
                    size = f"{downloaded / (1024 ** 3):.2f} GiB"
                    if total:
                        size += f" / {total / (1024 ** 3):.2f} GiB"
                    print(f"\rDownloading dataset: {size}", end="", flush=True)
                    last_report = now
    print()
    if total and downloaded != total:
        raise IOError(f"Download was incomplete: received {downloaded} of {total} bytes.")


def read_split(name: str) -> list[str]:
    url = SPLIT_URL.format(split=name)
    request = urllib.request.Request(url, headers={"User-Agent": "CropByte dataset setup"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return [
            line.strip()
            for line in response.read().decode("utf-8").splitlines()
            if line.strip()
        ]


def prepare_dataset() -> None:
    if DATA_DIR.exists():
        raise FileExistsError(
            f"{DATA_DIR} already exists. Move or rename it before preparing the dataset."
        )

    with (ROOT / "cropbyte_labels.json").open(encoding="utf-8") as labels_file:
        labels = json.load(labels_file)
    if set(labels) != set(SOURCE_CLASSES.values()):
        raise ValueError("cropbyte_labels.json does not match the downloader's class mapping.")

    split_paths = {
        "train": read_split("train"),
        "validation": read_split("test"),
    }
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cropbyte-plantvillage-") as temporary:
        temporary_dir = Path(temporary)
        archive_path = temporary_dir / "data.zip"
        download_file(ARCHIVE_URL, archive_path)
        prepared_dir = temporary_dir / "PlantVillage"
        counts = {split: {label: 0 for label in labels} for split in split_paths}

        with zipfile.ZipFile(archive_path) as archive:
            archive_names = set(archive.namelist())
            for split, paths in split_paths.items():
                for relative_path in paths:
                    image_path = PurePosixPath(relative_path)
                    if len(image_path.parts) != 4 or image_path.parts[:2] != ("raw", "color"):
                        continue
                    target_label = SOURCE_CLASSES.get(image_path.parts[2])
                    if target_label is None:
                        continue
                    archive_name = image_path.as_posix()
                    if archive_name not in archive_names:
                        raise FileNotFoundError(f"Image listed in the official split is missing: {archive_name}")

                    destination = prepared_dir / split / target_label / image_path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(archive_name) as source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output, length=CHUNK_SIZE)
                    counts[split][target_label] += 1

        for split, class_counts in counts.items():
            missing = [label for label, count in class_counts.items() if count == 0]
            if missing:
                raise ValueError(f"No {split} images found for these classes: {missing}")

        prepared_dir.replace(DATA_DIR)

    print(f"Prepared dataset at {DATA_DIR}")
    for split, class_counts in counts.items():
        print(f"{split}: {sum(class_counts.values())} images across {len(class_counts)} classes")


if __name__ == "__main__":
    prepare_dataset()
