
import re
import io
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ultralytics import YOLO
from PIL import Image


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DL_MODEL_PATH = BASE_DIR / "DL" / "models" / "MedEquip30_Model" / "best.pt"

NLP_MODEL_PATH = (
    BASE_DIR / "NLP" / "models" / "medical_specialty_model.pkl"
)

TFIDF_PATH = (
    BASE_DIR / "NLP" / "models" / "tfidf_vectorizer.pkl"
)

LABEL_ENCODER_PATH = (
    BASE_DIR / "NLP" / "models" / "label_encoder.pkl"
)


# ==========================================
# 2. GLOBAL MODEL VARIABLES
# ==========================================

dl_model = None
nlp_model = None
tfidf_vectorizer = None
label_encoder = None

stop_words = None
lemmatizer = None


# ==========================================
# 3. NLP RESOURCE SETUP
# ==========================================

def setup_nltk():

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4")
    ]

    for resource_path, download_name in resources:

        try:
            nltk.data.find(resource_path)

        except LookupError:

            success = nltk.download(
                download_name,
                quiet=True
            )

            if not success:
                raise RuntimeError(
                    f"Could not download NLTK resource: {download_name}"
                )


# ==========================================
# 4. LOAD BOTH TRAINED MODELS
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global dl_model
    global nlp_model
    global tfidf_vectorizer
    global label_encoder
    global stop_words
    global lemmatizer

    print("Loading AI Hospital DL and NLP models...")

    # Check model files

    required_files = [
        DL_MODEL_PATH,
        NLP_MODEL_PATH,
        TFIDF_PATH,
        LABEL_ENCODER_PATH
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Model file not found: {file_path}"
            )

    # Setup NLTK

    setup_nltk()

    # Load DL model

    dl_model = YOLO(str(DL_MODEL_PATH))

    print("YOLOv8 model loaded successfully!")

    # Load NLP model files

    nlp_model = joblib.load(NLP_MODEL_PATH)

    tfidf_vectorizer = joblib.load(TFIDF_PATH)

    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    print("NLP model loaded successfully!")

    # Verify feature compatibility

    expected_features = nlp_model.n_features_in_

    actual_features = len(tfidf_vectorizer.vocabulary_)

    if expected_features != actual_features:

        raise RuntimeError(
            "NLP model and TF-IDF feature dimensions do not match."
        )

    # Setup preprocessing objects

    stop_words = set(stopwords.words("english"))

    # Preserve medical negation words as in training

    medical_negations = {
        "no",
        "not",
        "nor",
        "never",
        "without",
        "neither",
        "none"
    }

    stop_words = stop_words - medical_negations

    lemmatizer = WordNetLemmatizer()

    print("All DL and NLP models loaded successfully!")

    yield

    print("AI Hospital API shutting down.")


# ==========================================
# 5. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="AI Hospital Intelligence API",
    description=(
        "Combined Deep Learning and NLP API "
        "for medical equipment detection "
        "and medical specialty classification."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ==========================================
# 6. CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# 7. NLP TEXT PREPROCESSING
# ==========================================

def preprocess_text(text: str):

    # 1. Convert text to lowercase

    text = text.lower()

    # 2. Remove punctuation and special characters

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text
    )

    # 3. Remove extra spaces

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # 4. Tokenization

    tokens = word_tokenize(text)

    # 5. Remove abnormal words

    tokens = [
        word for word in tokens
        if word.isalpha() and len(word) > 1
    ]

    # 6. Remove stopwords

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # 7. Lemmatization

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    # 8. Join tokens

    return " ".join(tokens)


# ==========================================
# 8. REQUEST MODEL FOR NLP
# ==========================================

class MedicalTextRequest(BaseModel):

    transcription: str = Field(
        ...,
        min_length=1,
        description="Medical transcription text"
    )


# ==========================================
# 9. GET API - ROOT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Welcome to AI Hospital Intelligence API",

        "status": "running",

        "models": [
            "YOLOv8 Medical Equipment Detection",
            "Linear SVM Medical Specialty Classification"
        ],

        "endpoints": {
            "health": "/health",
            "equipment_detection": "/predict/equipment",
            "medical_text": "/predict/medical-text",
            "documentation": "/docs"
        }
    }


# ==========================================
# 10. GET API - HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": (
            "healthy"
            if all([
                dl_model is not None,
                nlp_model is not None,
                tfidf_vectorizer is not None,
                label_encoder is not None
            ])
            else "not_ready"
        ),

        "dl_model_loaded": dl_model is not None,

        "nlp_model_loaded": nlp_model is not None,

        "tfidf_loaded": tfidf_vectorizer is not None,

        "label_encoder_loaded": label_encoder is not None
    }


# ==========================================
# 11. POST API - DL EQUIPMENT DETECTION
# ==========================================

@app.post("/predict/equipment")
async def predict_equipment(
    file: UploadFile = File(...)
):

    if dl_model is None:

        raise HTTPException(
            status_code=503,
            detail="DL model is not loaded."
        )

    # Validate uploaded image

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ]:

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    try:

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        if len(image_bytes) > 10 * 1024 * 1024:

            raise HTTPException(
                status_code=413,
                detail="Image size must not exceed 10 MB."
            )

        # Convert uploaded image into RGB

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # YOLOv8 prediction

        results = dl_model.predict(
            source=image,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        detections = []

        # Extract detection results

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0].item())

                confidence = float(
                    box.conf[0].item()
                )

                class_name = dl_model.names[class_id]

                coordinates = [
                    round(float(value), 2)
                    for value in box.xyxy[0].tolist()
                ]

                detections.append({

                    "equipment": class_name,

                    "confidence": round(
                        confidence,
                        4
                    ),

                    "bounding_box": coordinates

                })

        return {

            "success": True,

            "model": "YOLOv8",

            "filename": file.filename,

            "total_detections": len(detections),

            "detections": detections

        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Equipment detection failed: {str(error)}"
        )

    finally:

        await file.close()


# ==========================================
# 12. POST API - NLP MEDICAL TEXT CLASSIFICATION
# ==========================================

@app.post("/predict/medical-text")
def predict_medical_text(
    request: MedicalTextRequest
):

    if any([
        nlp_model is None,
        tfidf_vectorizer is None,
        label_encoder is None
    ]):

        raise HTTPException(
            status_code=503,
            detail="NLP model is not loaded."
        )

    try:

        transcription = request.transcription.strip()

        if not transcription:

            raise HTTPException(
                status_code=400,
                detail="Please enter medical transcription text."
            )

        # Step 1: Preprocess text

        cleaned_text = preprocess_text(
            transcription
        )

        if not cleaned_text:

            raise HTTPException(
                status_code=400,
                detail="No valid words found after preprocessing."
            )

        # Step 2: TF-IDF transformation

        text_vectorized = tfidf_vectorizer.transform(
            [cleaned_text]
        )

        # Step 3: Linear SVM prediction

        prediction = nlp_model.predict(
            text_vectorized
        )

        # Step 4: Decode predicted label

        predicted_specialty = label_encoder.inverse_transform(
            prediction
        )[0]

        return {

            "success": True,

            "model": "LinearSVC",

            "predicted_medical_specialty": str(
                predicted_specialty
            ).strip(),

            "message": (
                "Medical specialty classification completed."
            )

        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Medical text prediction failed: {str(error)}"
        )


# ==========================================
# 13. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
