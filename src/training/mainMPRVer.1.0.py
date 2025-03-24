import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso

import sys
sys.stdout.reconfigure(encoding='utf-8')

# 讀取水質資料
dataset = pd.read_csv('data/dataV1_50000.csv')
print(dataset)
df = pd.DataFrame(dataset)

# 以平均值填補缺失值
imputer = SimpleImputer(strategy='mean')
data = imputer.fit_transform(df)

# 將填補好的水質資料轉換為 DataFrame 格式
data = pd.DataFrame(np.round(data, 2), columns=df.columns)

# 切分資料集
X = data.drop(columns=['Score'])
y = data['Score'].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# 創建隨機森林迴歸模型
model = Pipeline([
    ("poly_features", PolynomialFeatures()),
    ("std_scaler", StandardScaler()),
    ('lasso', Lasso())
])

# 網格搜索參數
param_grid = {
    'poly_features__degree': [1, 2, 3, 4, 5, 6]
}

# 交叉驗證
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# 網格搜索
grid_search.fit(X_train, y_train)

# 獲取最佳參數和最佳得分
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("最佳參數：", best_params, "最佳得分：", best_score)

# 使用最佳參數構建模型
model.set_params(**best_params)
model.fit(X_train, y_train)

# 預測結果
y_pred = model.predict(X_test)

# 計算模型的性能
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)
print("Train R2: ", train_r2)
print("Test R2: ", test_r2)

train_rmse = root_mean_squared_error(y_train, train_preds)
test_rmse = root_mean_squared_error(y_test, test_preds)
print("Train RMSE: ", train_rmse)
print("Test RMSE: ", test_rmse)

train_mae = mean_absolute_error(y_train, train_preds)
test_mae = mean_absolute_error(y_test, test_preds)
print("Train MAE: ", train_mae)
print("Test MAE: ", test_mae)

# 查看預測結果
print(y_pred)

# 保存模型
joblib.dump(model, 'modelMPRVer.1.0-50000.pkl')
