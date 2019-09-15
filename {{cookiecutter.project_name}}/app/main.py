from fastapi import FastAPI

from app.predict.views import urls as PredictView

app = FastAPI()
error = {404: {"description":"Not Found"}}

app.include_router(
    PredictView,
    tags = ['predict'],
    responses = error 
)