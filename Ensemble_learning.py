# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 09:09:03 2020

@author: C09700
"""
#%%
import numpy as np
import pandas as pd
from sklearn import ensemble, preprocessing, metrics
from sklearn.model_selection import train_test_split

# 載入資料
url = ".\data\Titanic.csv"
titanic_train = pd.read_csv(url)

# 填補遺漏值
age_median = np.nanmedian(titanic_train["Age"])

#np.where(condition, x, y)
#滿足條件(condition)，輸出x，不滿足輸出y。

new_Age = np.where(titanic_train["Age"].isnull(), age_median, titanic_train["Age"])
titanic_train["Age"] = new_Age

# 創造 dummy variables
label_encoder = preprocessing.LabelEncoder()
encoded_Sex = label_encoder.fit_transform(titanic_train["Sex"])

# 建立訓練與測試資料
# 轉置資料transpose
titanic_X = pd.DataFrame([
    titanic_train["Pclass"],
    encoded_Sex,
    titanic_train["Age"]
    ]).T

titanic_y = titanic_train["Survived"]
train_X, test_X, train_y, test_y = train_test_split(titanic_X, titanic_y, test_size = 0.3)

# 建立 bagging 模型
bag = ensemble.BaggingClassifier(n_estimators = 100)
bag_fit = bag.fit(train_X, train_y)

# 預測
test_y_predicted = bag.predict(test_X)

accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)


#%%
# 建立 boosting 模型
boost = ensemble.AdaBoostClassifier(n_estimators = 100)
boost_fit = boost.fit(train_X, train_y)

# 預測
test_y_predicted = boost.predict(test_X)

# 績效
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)
#%%