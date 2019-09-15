from fastapi import APIRouter, File, UploadFile

from app.predict.service import prediction

urls = APIRouter()

@urls.post('/predict')
async def predict_image(file: UploadFile = File(...)):
    img_bytes = file.read()
    class_id, class_name = prediction.predict(image_bytes=img_bytes)
    return {
        "status" : "success",
        "content" : {
            "predict_name" : f"{class_name}",
            "predict_id" : f"{class_id}"
            }
        }