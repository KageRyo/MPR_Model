import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
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

X = data[[0,1,2,3,4]]
y = data[[5]]
y = y.values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# 創建一個多元多項式回歸模型
model = Pipeline([
    ("poly_features", PolynomialFeatures()),
    ("std_scaler", StandardScaler()),
    ('lasso', Lasso())
])
param_grid = {
    'poly_features__degree': [4,5,6,7],      # 多項式階數 5
    'lasso__alpha': [0.45,0.46,0.47,0.48,0.49,0.5,]  # 正規化參數 0.46
}
# 創建 GridSearchCV 對象，並進行交叉驗證
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
rmse = mean_squared_error(y_test, y_pred)**0.5

# 查看預測結果
print(y_pred)
print("R方：", r2)
print("RMSE：", rmse)

# 可視化
fig = plt.figure(figsize=(10, 6))
t = fig.suptitle('Polynomial Regression', fontsize=14)
ax = fig.add_subplot(111, projection='3d')

#使用 Salary 列作為顏色映射的依據
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['green','yellow','orange', 'red'])
scatter = ax.scatter(data[1], data[2], data[3], c=data[4], cmap=cmap, s=40, marker='o', edgecolors='none', alpha=0.8)

# 設置坐標軸標籤
ax.set_xlabel('DO')
ax.set_ylabel('BOD')
ax.set_zlabel('NH3-N')

# 創建颜色條
cb = plt.colorbar(scatter,pad=0.1)
cb.set_label('SS')

# 保存為png
plt.savefig('test.png')

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# 保存模型
joblib.dump(model, 'modelVer.1.03.pkl')


# 顯示圖形
plt.show()