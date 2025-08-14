import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Tuple, List

# =============================
# 🔧 CONFIGURE THESE CONSTANTS
# =============================
# If you know your class names, put them here in the correct order.
# Otherwise, the app will generate generic labels like "class_0, class_1, ..." at runtime.
CLASS_NAMES: List[str] = []  # e.g., ["cat", "dog", "rabbit"]

# Path to your Keras .h5 model file. You can also set via Streamlit sidebar.
DEFAULT_MODEL_PATH = r"C:\Users\hp\Downloads\cropbyte_model.h5"

# If your model expects a specific input size, you can leave this as None to auto-detect
# from model.input_shape at runtime. If auto-detection fails, we'll fall back to (224, 224).
FALLBACK_INPUT_SIZE: Tuple[int, int] = (224, 224)

# ======================================
# 🌐 Translations for 4 UI languages
# ======================================
T = {
    "en": {
        "lang_name": "English",
        "app_title": "CropByte",
        "subtitle": "Select language, upload an image, and get predictions.",
        "model_path": "Model (.h5) path",
        "load_model": "Load model",
        "model_loaded": "Model loaded successfully.",
        "model_failed": "Failed to load model. Check the path and file.",
        "upload": "Upload an image",
        "predict": "Predict",
        "predictions": "Predictions",
        "topk": "Top-K (1-10)",
        "prob": "Probability",
        "class": "Class",
        "no_image": "Please upload an image to run prediction.",
        "footer": "Built with Streamlit + TensorFlow",
        "sidebar_title": "Settings",
        "choose_lang": "Language",
    },
    "hi": {
        "lang_name": "हिंदी",
        "app_title": "क्रॉपबाइट",  # Hindi transliteration
        "subtitle": "भाषा चुनें, छवि अपलोड करें और भविष्यवाणी देखें।",
        "model_path": "मॉडल (.h5) का पथ",
        "load_model": "मॉडल लोड करें",
        "model_loaded": "मॉडल सफलतापूर्वक लोड हो गया।",
        "model_failed": "मॉडल लोड नहीं हो पाया। पथ और फ़ाइल जाँचें।",
        "upload": "छवि अपलोड करें",
        "predict": "भविष्यवाणी करें",
        "predictions": "भविष्यवाणियाँ",
        "topk": "शीर्ष-K (1-10)",
        "prob": "संभावना",
        "class": "वर्ग",
        "no_image": "कृपया भविष्यवाणी के लिए छवि अपलोड करें।",
        "footer": "Streamlit + TensorFlow से निर्मित",
        "sidebar_title": "सेटिंग्स",
        "choose_lang": "भाषा",
    },
    "ml": {
        "lang_name": "മലയാളം",
        "app_title": "ക്രോപ്ബൈറ്റ്",  # Malayalam transliteration
        "subtitle": "ഭാഷ തിരഞ്ഞെടുക്കുക, ചിത്രം അപ്‌ലോഡ് ചെയ്യുക, പ്രവചനം കാണുക.",
        "model_path": "മോഡൽ (.h5) പാത",
        "load_model": "മോഡൽ ലോഡ് ചെയ്യുക",
        "model_loaded": "മോഡൽ വിജയകരമായി ലോഡ് ചെയ്തു.",
        "model_failed": "മോഡൽ ലോഡ് ചെയ്യാൻ കഴിഞ്ഞില്ല. പാത്തും ഫയലും പരിശോധിക്കുക.",
        "upload": "ചിത്രം അപ്‌ലോഡ് ചെയ്യുക",
        "predict": "പ്രവചിക്കുക",
        "predictions": "പ്രവചനങ്ങൾ",
        "topk": "ടോപ്പ്-K (1-10)",
        "prob": "സാദ്ധ്യത",
        "class": "വർഗം",
        "no_image": "പ്രവചനത്തിനായി ദയവായി ചിത്രം അപ്‌ലോഡ് ചെയ്യുക.",
        "footer": "Streamlit + TensorFlow ഉപയോഗിച്ച് നിർമ്മിച്ചത്",
        "sidebar_title": "ക്രമീകരണങ്ങൾ",
        "choose_lang": "ഭാഷ",
    },
    "ta": {
        "lang_name": "தமிழ்",
        "app_title": "க்ராப்பைட்",  # Tamil transliteration
        "subtitle": "மொழியைத் தேர்ந்தெடுத்து, படத்தை பதிவேற்றி, கணிப்பைப் பெறுங்கள்.",
        "model_path": "மாடல் (.h5) பாதை",
        "load_model": "மாடலை ஏற்று",
        "model_loaded": "மாடல் வெற்றிகரமாக ஏற்றப்பட்டது.",
        "model_failed": "மாடலை ஏற்ற முடியவில்லை. பாதையும் கோப்பையும் சரிபார்க்கவும்.",
        "upload": "படத்தை பதிவேற்றவும்",
        "predict": "கணிக்க",
        "predictions": "கணிப்புகள்",
        "topk": "டாப்-K (1-10)",
        "prob": "சாத்தியம்",
        "class": "வகை",
        "no_image": "கணிப்புக்கு தயவுசெய்து ஒரு படத்தை பதிவேற்றவும்.",
        "footer": "Streamlit + TensorFlow மூலம் உருவாக்கப்பட்டது",
        "sidebar_title": "அமைப்புகள்",
        "choose_lang": "மொழி",
    },
}

LANG_OPTIONS = {
    "English": "en",
    "हिंदी": "hi",
    "മലയാളം": "ml",
    "தமிழ்": "ta",
}

# =============================
# 🧠 Utilities
# =============================
@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
    return tf.keras.models.load_model(path)

@st.cache_resource(show_spinner=False)
def get_input_size(model) -> Tuple[int, int]:
    try:
        ishape = model.input_shape
        # input_shape could be (None, H, W, C) or list of shapes
        if isinstance(ishape, list):
            ishape = ishape[0]
        if len(ishape) == 4:
            h, w = int(ishape[1]), int(ishape[2])
            return (h, w)
    except Exception:
        pass
    return FALLBACK_INPUT_SIZE


def preprocess_image(img: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    img = img.convert("RGB").resize((target_size[1], target_size[0]))  # (W, H)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def get_class_names(num_logits: int) -> List[str]:
    if CLASS_NAMES and len(CLASS_NAMES) == num_logits:
        return CLASS_NAMES
    # Fallback generic labels
    return [f"class_{i}" for i in range(num_logits)]


# =============================
# 🎛️ Sidebar: language + settings
# =============================
st.set_page_config(page_title="Multilingual Image Classifier", page_icon="🖼️", layout="centered")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    lang_readable = st.selectbox("Language / भाषा / ഭാഷ / மொழி", list(LANG_OPTIONS.keys()), index=0)
    lang = LANG_OPTIONS[lang_readable]
    txt = T[lang]

    st.markdown("---")
    model_path = st.text_input(txt["model_path"], value=DEFAULT_MODEL_PATH)
    load_btn = st.button(txt["load_model"])  

# =============================
# 🖥️ Main UI
# =============================
st.title(T[lang]["app_title"])
st.caption(T[lang]["subtitle"])  

# Load / cache model on-click so changing path is respected
if "_model" not in st.session_state or load_btn or st.session_state.get("_model_path") != model_path:
    try:
        with st.spinner("Loading model…"):
            model = load_keras_model(model_path)
        st.session_state["_model"] = model
        st.session_state["_model_path"] = model_path
        st.success(T[lang]["model_loaded"], icon="✅")
    except Exception as e:
        st.session_state["_model"] = None
        st.error(f"{T[lang]['model_failed']}\n\n{e}")

model = st.session_state.get("_model")

uploaded = st.file_uploader(T[lang]["upload"], type=["jpg", "jpeg", "png", "webp"])

# Remove Top-K slider and set topk to 1
topk = 1
predict_clicked = st.button(T[lang]["predict"], type="primary", use_container_width=True)

# Display preview
if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Preview", use_column_width=True)

# Predict
if predict_clicked:
    if model is None:
        st.error(T[lang]["model_failed"])  # Model not loaded
    elif uploaded is None:
        st.warning(T[lang]["no_image"]) 
    else:
        try:
            input_size = get_input_size(model)
            x = preprocess_image(img, input_size)
            preds = model.predict(x)
            preds = np.array(preds)
            if preds.ndim == 1:
                preds = np.expand_dims(preds, 0)
            # If model outputs logits, convert to probabilities
            if preds.shape[-1] > 1:
                probs = tf.nn.softmax(preds, axis=-1).numpy()[0]
            else:
                # Binary or single logit; map to probability via sigmoid
                probs = tf.nn.sigmoid(preds).numpy().flatten()
                if probs.size == 1:
                    probs = np.array([1 - probs[0], probs[0]], dtype=np.float32)
            class_names = get_class_names(len(probs))
            top_idx = np.argsort(probs)[::-1][:topk]

            st.subheader(T[lang]["predictions"])
            for rank, i in enumerate(top_idx, start=1):
                st.write(f"{rank}. **{class_names[i]}** — {probs[i]*100:.2f}%")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown(f"\n\n— {T[lang]['footer']}")
