from io import BytesIO
from PIL import Image

from fastapi import APIRouter, UploadFile

from ..classifier.model import predict_image

router = APIRouter()


@router.post("/classify-image")
async def classify_image(file: UploadFile):
    # Check if it has the correct extension
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return {"image_format": "Image must be jpg, jpeg, or png format!"}

    # Read the uploaded image
    image_contents = await file.read()

    # Open the image using PIL
    image = Image.open(BytesIO(image_contents))

    # Classify the disease
    disease_status = predict_image(image=image)

    return {"disease_status": disease_status}

