import pandas as pd
import joblib
from fastapi import FastAPI
from typing import Optional
from enum import Enum

app = FastAPI()

@app.get("/blog/pagination/{csv}")
def predict(csv:str):
    data=pd.read_csv(f'{csv}')
    df = pd.DataFrame(data)
    model=joblib.load('modelEXVer.6.01.pkl')
    predictions = model.predict(df)
    return {"message": f"預測結果：{predictions}"}
