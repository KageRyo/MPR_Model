import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor,GradientBoostingRegressor
import time

# 讀取數據
dataset=pd.read_csv('data/'+os.listdir('data')[0])
print(dataset)
df = pd.DataFrame(dataset)
#以平均值填補缺失值
imputer = SimpleImputer(strategy='mean') 
data = imputer.fit_transform(df)
#將填補好的數據轉換為DataFrame格式
data = pd.DataFrame(np.round(data, 2))
print(pd.DataFrame(data))
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
X = data[[0,1,2,3,4]]
y = data[[5]]
y = y.values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# 創建一個多元多項式回歸模型
poly_pipeline = Pipeline([
    ('poly_features', PolynomialFeatures()),  
    ('std_scaler', StandardScaler()),         
    ('lasso', Lasso())                        
])
# 創建其他基本模型的 Pipeline
rf_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
    ('rf', RandomForestRegressor())
])
gbt_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
    ('gbt', GradientBoostingRegressor())
])

# 創建 VotingRegressor 對象，將多項式迴歸模型和其他基本模型一起作為基本估計器
model = VotingRegressor(estimators=[
    ('poly', poly_pipeline),
    ('random_forest', rf_pipeline),
    ('gradient_boosting', gbt_pipeline)
])
param_grid = {
    'poly__poly_features__degree': [5,6,7,8],   # 多項式階數
    'poly__lasso__alpha': [0.1,0.5,1.0],      # Lasso 的正規化參數
    'random_forest__rf__n_estimators': [25,50,75,100],# 隨機森林中決策數的數量
    'gradient_boosting__gbt__n_estimators': [25,50,75,100] # 梯度提升樹中決策數的數量
}
# 創建 GridSearchCV 對象，並進行交叉驗證
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
# 在資料上執行網格搜索
grid_search.fit(X_train, y_train)
# 取得最佳參數組合和效能得分
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("最佳参数：", best_params, "最佳得分：", best_score)
# 使用最佳參數重新訓練模型
model.set_params(**best_params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 計算模型的性能
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5

# 查看預測結果
print(y_pred)
print("R方：", r2)
print("RMSE：", rmse)

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# 保存模型
joblib.dump(model, 'modelVer.6.01.pkl')

