# MPR_Model
MPR_Model 是基於機器學習的多元多項式水質分析模型，可幫助您快速分析 DO、BOD、NH3-N、EC 及 SS 等水質資料，並基於水質指標（Water Quality Index, WQI、WQI5）進行評估。  

# 如何使用
1. 首先您需要 Clone 此專案。  
```bash
git clone https://github.com/RotatingPotato/MPR_Model.git
```
2. 接著您必須安裝所需的函式庫。  
> 請確保您的裝置中有可使用的 Python 版本。  
> 當然，您也可以建立 venv 虛擬環境或使用 Conda 後再執行。  
```bash
pip install -r requirements.txt
```
3. 您可以在 `main.py` 中修改您希望使用的IP（預設直接使用 0.0.0.0）和端口（預設為 8000）。  
4. 進入 MPR_Model 的路徑後於 Terminal 中執行 MPR_Model。  
```bash
cd MPR_Model
python3 main.py
```

# 建議
- 本應用程式可接收之CSV格式如下：

| DO     	| BOD     	| NH3N     	| EC     	| SS     	|
|--------	|---------	|----------	|--------	|--------	|
| DO數值 	| BOD數值 	| NH3N數值 	| EC數值 	| SS數值 	|
| ...    	                                          	|
  
- 您可以搭配 [WaterMirror](https://github.com/RotatingPotato/WaterMirror) 作為前端行動裝置應用程式使用。

## 貢獻
歡迎貢獻！ 如果您發現任何錯誤或有改進建議，歡迎提出 Issues 或 Pull Requests。  

## LICENSE  
本程式開源授權條款討論中，請靜待開放，謝謝您。   
有任何疑問請洽 kageryo@coderyo.com  
+ 張健勳 Chien-Hsun, Chang [@KageRyo](https://github.com/KageRyo)   
+ 吳國維 Kuo-Wei, Wu [@RRAaru](https://github.com/RRAaru)

${{\color{orange}{\textsf{本作品為 張健勳 與 吳國維 用於「國立臺中科技大學智慧生產工程系」畢業專題之作品，其著作權由兩人共同擁有。}}}}\$  
${{\color{yellow}{\textsf{特別感謝「國立臺中科技大學智慧生產工程系」蔡文宗 教授指導。}}}}\$  
