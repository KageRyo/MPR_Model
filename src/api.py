import numpy as np
import pandas as pd
import joblib
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File, Query

app = FastAPI()

# 在記憶體中緩存整個母體資料集的分數
try:
    all_scores = pd.read_csv('data/model_data.csv')['Score']
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading data: {e}")

# 讀取模型
def load_model():
    try:
        return joblib.load('models/modelVer.1.4.1.pkl')
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

# 評估個別水質項目的品質
def assess_quality(df):
    assess_dict = {}
    # 定義水質項目的品質評估範圍
    def assess_range(val, good_rng, normal_rngs, bad_rng):
        if good_rng[0] <= val <= good_rng[1]:
            return '好'
        elif any(normal_rng[0] <= val <= normal_rng[1] for normal_rng in normal_rngs):
            return '普通'
        elif bad_rng[0] <= val <= bad_rng[1]:
            return '不好'
        else:
            return '異常'
    # 計算各項目的品質評估
    assess_dict['DO'] = assess_range(df['DO'].mean(), [80, 120], [[55, 80], [120, 140]], [0, 55])
    assess_dict['BOD'] = assess_range(df['BOD'].mean(), [0, 2], [[2, 4]], [4, 25])
    assess_dict['NH3N'] = assess_range(df['NH3N'].mean(), [0, 0.1], [[0.1, 1]], [1, 8])
    assess_dict['SS'] = assess_range(df['SS'].mean(), [0, 10], [[10, 40]], [40, 1000])
    assess_dict['EC'] = assess_range(df['EC'].mean(), [0, 400], [[400, 750]], [750, 3000])
    return assess_dict

# API : 測試 API 是否正常運作 (GET)
# http://<apiurl>:8000/
@app.get("/")
async def read_root():
    return {"message": "成功與 API 連線!"}

# API：計算指定分數的百分位數 (GET)
# http://<apiurl>:8000/percentile/?score=${data}
@app.get("/percentile/")
async def calculate_percentile(score: float = Query(..., description="The score to evaluate")):
    try:
        percentile = (all_scores <= score).mean() * 100
        return {"percentile": percentile}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating percentile: {e}")
    
# API : 計算分數的類別 (GET)
# http://<apiurl>:8000/categories/
@app.get("/categories/")
async def get_categories():
    try:
        df = pd.read_csv('data/model_data.csv')
        bins = [0, 15, 30, 50, 70, 85, 100]  # 定義分數範圍的界限
        labels = ['惡劣', '糟糕', '不良', '中等', '良好', '優良']
        df['Category'] = pd.cut(df['Score'], bins=bins, labels=labels, right=False)
        category_counts = df['Category'].value_counts().reset_index()
        category_counts.columns = ['category', 'rating']
        return {"data": category_counts.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving categories: {e}")

# API：水質資料分析每筆資料平均總分數 (POST)
# http://<apiurl>:8000/score/total/
@app.post("/score/total/") # 使用 POST 方法處理上傳的 CSV 檔案
async def predict_total(file: UploadFile = File(...)):
    try:
        print("收到前端POST請求")
        df = pd.read_csv(file.file)
        df['DO_BOD_ratio'] = np.where(df['BOD'] == 0, 0, df['DO'] / df['BOD'])
        df['BOD_NH3N_product'] = df['BOD'] * df['NH3N']
        print(df)
        predictions = predict_scores(df)        # AI機器學習（MPR模型）進行分析
        result = float(np.mean(predictions))    # 計算平均分數
        assessment = assess_quality(df)         # 進行個別項目品質評估
        print(f"Received file: {file.filename}")
        print(f"Content type: {file.content_type}")
        print(result)
        print(assessment)
        return {"average_score": result, "assessment": assessment}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {e}")

# API：水質資料分析每筆資料分數 (POST)
# http://<apiurl>:8000/score/all/
@app.post("/score/all/") # 使用 POST 方法處理上傳的 CSV 檔案
async def predict_all(file: UploadFile = File(...)):
    try:
        print("收到前端POST請求")
        df = pd.read_csv(file.file)
        df['DO_BOD_ratio'] = np.where(df['BOD'] == 0, 0, df['DO'] / df['BOD'])
        df['BOD_NH3N_product'] = df['BOD'] * df['NH3N']
        print(df)
        predictions = predict_scores(df)        # AI機器學習（MPR模型）進行分析
        result = float(np.mean(predictions))    # 計算平均分數
        assessment = assess_quality(df)         # 進行個別項目品質評估
        print(f"Received file: {file.filename}")
        print(f"Content type: {file.content_type}")
        print(result)
        print(assessment)
        return {"average_score": result, "assessment": assessment}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {e}")

# API程式起始點
if __name__ == "__main__":
    app.run()