from fastapi import FastAPI
from predictor import predict_flower
from pydantic import BaseModel
import pandas as pd

flowerAPI = FastAPI()

class Flower(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@flowerAPI.get('/')
async def root():
    return {'message': 'This API Predicts Flower Type'}

@flowerAPI.post('/predict')
async def getLabel(flower : Flower):
    data = flower
    df = pd.DataFrame(
        [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]],
        columns=['sepal length (cm)', 'sepal width (cm)','petal length (cm)','petal width (cm)']
        )
    label = predict_flower(df)
    return {'Label':label}

    



