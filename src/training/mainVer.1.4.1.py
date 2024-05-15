import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer

# 讀取數據
dataset=pd.read_csv('data/'+os.listdir('data')[6])
print(dataset)
df = pd.DataFrame(dataset)
#以平均值填補缺失值
imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(df)
# 將填補好的數據轉換為DataFrame格式
data = pd.DataFrame(np.round(data, 2), columns=df.columns)
# 創建新特徵
data['DO_BOD_ratio'] = np.where(data['BOD'] == 0, 0, data['DO'] / data['BOD'])
data['BOD_NH3N_product'] = data['BOD'] * data['NH3N']
# 切分數據集
X = data.drop(columns=['Score'])
y = data['Score'].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# 創建多元多項式回歸模型
model = Pipeline([
    ("poly_features", PolynomialFeatures()),  
    ("std_scaler", StandardScaler()),
    ('lasso', Lasso())  
])
# 網格搜索參數
param_grid = {
    'poly_features__degree': [6],      
    'lasso__alpha': [0.1], 
}
# 交叉驗證
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
# 網格搜索
grid_search.fit(X_train, y_train)
# 獲取最佳參數和最佳得分
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("最佳参数：", best_params, "最佳得分：", best_score)
# 使用最佳參數構建模型
model.set_params(**best_params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 計算模型的性能
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5

# 查看預測結果
print(y_pred)
print("R方：", r2)
print("MAE：", mae)
print("RMSE：", rmse)

# 保存模型
joblib.dump(model, 'modelVer.1.4.1.pkl')