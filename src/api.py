import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File

app = FastAPI()

# 讀取模型
def load_model():
    try:
        return joblib.load('models/modelVer.6.01.pkl')
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

# 讀取分析資料
def load_data(csv):
    try:
        return pd.read_csv(f'data/{csv}')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Data file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {e}")

# 進行分析與評估
def predict_scores(df):
    model = load_model()
    predictions = np.round(model.predict(df), 3)
    return predictions

# API：水質資料分析每筆資料平均總分數
@app.post("/score/total/") # 使用 POST 方法處理上傳的 CSV 檔案
async def predict_total(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        predictions = predict_scores(df)
        return float(np.mean(predictions))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {e}")

# API：水質資料分析每筆資料分數
@app.post("/score/all/") # 使用 POST 方法處理上傳的 CSV 檔案
async def predict_all(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        predictions = predict_scores(df)
        return predictions.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {e}")

if __name__ == "__main__":
    app.run()