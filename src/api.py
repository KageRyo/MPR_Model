import pandas as pd
import joblib
import os
from fastapi import FastAPI
from typing import Optional
from enum import Enum

app = FastAPI()

@app.get("/blog/pagination/{csv}")
def predict(csv:str):
    current_dir = os.getcwd()
    dataPath = os.path.join(current_dir, "data", csv)
    data=pd.read_csv(dataPath)
    df = pd.DataFrame(data)
    model=joblib.load('models\modelVer.6.01.pkl')
    predictions = model.predict(df)
    return {"message": f"預測結果：{predictions}"}