from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import io
from model import TumorCNN

# =========================
# CONFIG
# =========================
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# INIT APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL
# =========================
model = TumorCNN()
model.load_state_dict(torch.load("best_brain_tumor_model.pth", map_location=device))
model.to(device)
model.eval()

# =========================
# TRANSFORMS
# =========================
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# PREPROCESSING
# =========================
def mri_preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        image = image[y:y+h, x:x+w]

    gray_cropped = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_cropped)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

def preprocess_image(image):
    image = np.array(image)
    image = mri_preprocessing(image)
    image = Image.fromarray(image)
    image = val_transforms(image)
    image = image.unsqueeze(0)
    return image.to(device)

# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return {"message": "Brain Tumor API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        return {"error": "Invalid image"}

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": float(confidence)
    }
