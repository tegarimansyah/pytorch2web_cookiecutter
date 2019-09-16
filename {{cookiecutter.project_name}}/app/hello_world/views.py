from fastapi import APIRouter

urls = APIRouter()

@urls.get('/')
async def predict_image():
    return {
        "status" : "success",
        "content" : {
            "msg": "Hello World!"
            }
        }