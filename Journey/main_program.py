import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import kagglehub
#导入相关的库


train = pd.read_csv("/kaggle/input/competitions/home-data-for-ml-course/train.csv",index_col=0)
test = pd.read_csv('/kaggle/input/competitions/home-data-for-ml-course/test.csv', index_col=0)

X = pd.concat([train.drop("SalePrice",axis=1),test],axis=0)
y = train["SalePrice"]
#读取数据，X合并train与test方便后面研究处理数据
X.info()


from statsmodels.stats.outliers_influence import variance_inflation_factor
#研究多重共线性的库
from sklearn.impute import SimpleImputer
#填充器的库，方便缺失值处理

import seaborn as sns
import matplotlib.pyplot as plt
#画图工具

sns.set_style("darkgrid")
#设置基调为灰色网格

#数据研究与处理

X.drop(['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageCars'],axis=1,inplace=True)
#去掉多重共线性的变量

X.drop(['PoolQC','MiscFeature','Alley'], axis=1, inplace=True)
#去掉缺失值严重的变量

X.drop(['MoSold','YrSold'],axis=1,inplace=True)
#去除无意义的，相关性不显著的变量

cat_col = X.select_dtypes(include=['object']).columns
overfit_cat = []
for i in cat_col:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 96:
        overfit_cat.append(i)

overfit_cat = list(overfit_cat)
X = X.drop(overfit_cat, axis=1)
#去掉单一过多的字符变量，对预测意义不大

num_col = X.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).columns
overfit_num = []
for i in num_col:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 96:
        overfit_num.append(i)

overfit_num = list(overfit_num)
X = X.drop(overfit_num, axis=1)
#去掉单一过多的数字变量，对预测意义不大

train = train.drop(train[train['LotFrontage'] > 200].index)
train = train.drop(train[train['LotArea'] > 100000].index)
train = train.drop(train[train['BsmtFinSF1'] > 4000].index)
train = train.drop(train[train['TotalBsmtSF'] > 5000].index)
train = train.drop(train[train['GrLivArea'] > 4000].index)
#去除异常值

#对缺失值处理

cat = ['GarageType','GarageFinish','BsmtFinType2','BsmtExposure','BsmtFinType1', 
       'GarageCond','GarageQual','BsmtCond','BsmtQual','FireplaceQu','Fence',"KitchenQual",
       "HeatingQC",'ExterQual','ExterCond']

X[cat] = X[cat].fillna("NA")
#对变量研究，该字符变量的缺失值对分析有用，直接使用NA


cols = ["MasVnrType", "MSZoning", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "Functional"]
X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0]))
#该变量大多是以街区为单位的分布，故取街区众数代替


X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
X['GarageArea'] = X.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.mean()))
X['MSZoning'] = X.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
#处理同上，以某个变量为单位的分布，取众数代表大多如此，取平均数代表无异常值影响，取平均程度

cont = ["BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea"]
X[cont] = X[cont].fillna(X[cont].mean())
#对连续变量，且无极端值，直接取平均值

X['MSSubClass'] = X['MSSubClass'].apply(str)
#转换为字符类型

ordinal_map = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA':0}
fintype_map = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'NA': 0}
expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
fence_map = {'GdPrv': 4,'MnPrv': 3,'GdWo': 2, 'MnWw': 1,'NA': 0}
#这类字符代表了一种排名的优劣，直接对应，方便机器学习，以预测SalePrice

ord_col = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond', 'FireplaceQu']
for col in ord_col:
    X[col] = X[col].map(ordinal_map)
    
fin_col = ['BsmtFinType1','BsmtFinType2']
for col in fin_col:
    X[col] = X[col].map(fintype_map)

X['BsmtExposure'] = X['BsmtExposure'].map(expose_map)
X['Fence'] = X['Fence'].map(fence_map)
#数据map对应处理


X['TotalLot'] = X['LotFrontage'] + X['LotArea']
X['TotalBsmtFin'] = X['BsmtFinSF1'] + X['BsmtFinSF2']
X['TotalSF'] = X['TotalBsmtSF'] + X['2ndFlrSF']
X['TotalBath'] = X['FullBath'] + X['HalfBath']
X['TotalPorch'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['ScreenPorch']
#相加提供总量以便更好预测，而原来的数据提供细节

colum = ['MasVnrArea','TotalBsmtFin','TotalBsmtSF','2ndFlrSF','WoodDeckSF','TotalPorch']

for col in colum:
    col_name = col+'_bin'
    X[col_name] = X[col].apply(lambda x: 1 if x > 0 else 0)
#进行特征的二值化处理，关注是否存在，原值提供细节。在这是否存在更重要


X = pd.get_dummies(X)
#独热编码

y["SalePrice"] = np.log(y['SalePrice'])
#确保类似正态分布

#模型准备训练
x = X.loc[train.index]
y = y.loc[train.index]
test = X.loc[test.index]


from sklearn.preprocessing import RobustScaler
#防止异常值干扰

cols = x.select_dtypes(np.number).columns
transformer = RobustScaler().fit(x[cols])
x[cols] = transformer.transform(x[cols])
test[cols] = transformer.transform(test[cols])

from sklearn.model_selection import train_test_split
#划分

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=2020)

from sklearn.metrics import mean_squared_error, mean_absolute_error
#结果评价
from xgboost import XGBRegressor
#一个模型
from sklearn import ensemble
#集成学习：随机森林，梯度提升决策树，AdaBoost
from lightgbm import LGBMRegressor
#一个模型
from sklearn.model_selection import cross_val_score
#结果评价


#开始训练
xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror')
#使用梯度提升树，与均方根方差

from sklearn.model_selection import RandomizedSearchCV
#RandomizedSearCV: 通过随机搜索和交叉验证优化模型的超参数

param_lst = {
    'learning_rate' : [0.01, 0.1, 0.15, 0.3, 0.5],
    'n_estimators' : [100, 500, 1000, 2000, 3000],
    'max_depth' : [3, 6, 9],
    'min_child_weight' : [1, 5, 10, 20],
    'reg_alpha' : [0.001, 0.01, 0.1],
    'reg_lambda' : [0.001, 0.01, 0.1]
}

xgb_reg = RandomizedSearchCV(estimator = xgb, param_distributions = param_lst,
                              n_iter = 100, scoring = 'neg_root_mean_squared_error',
                              cv = 5)
#在上述组合中随机选100组，并进行5交叉验证

       
xgb_search = xgb_reg.fit(X_train, y_train)
#开始训练

best_param = xgb_search.best_params_
xgb = XGBRegressor(**best_param)
#解包最优参数


#开始训练
lgbm = LGBMRegressor(boosting_type='gbdt',objective='regression', max_depth=-1,
                    lambda_l1=0.0001, lambda_l2=0, learning_rate=0.1,
                    n_estimators=100, max_bin=200, min_child_samples=20, 
                    bagging_fraction=0.75, bagging_freq=5,
                    bagging_seed=7, feature_fraction=0.8,
                    feature_fraction_seed=7, verbose=-1)

param_lst = {
    'max_depth' : [2, 5, 8, 10],
    'learning_rate' : [0.001, 0.01, 0.1, 0.2],
    'n_estimators' : [100, 300, 500, 1000, 1500],
    'lambda_l1' : [0.0001, 0.001, 0.01],
    'lambda_l2' : [0, 0.0001, 0.001, 0.01],
    'feature_fraction' : [0.4, 0.6, 0.8],
    'min_child_samples' : [5, 10, 20, 25]
}

lightgbm = RandomizedSearchCV(estimator = lgbm, param_distributions = param_lst,
                              n_iter = 100, scoring = 'neg_root_mean_squared_error',
                              cv = 5)
       
lightgbm_search = lightgbm.fit(X_train, y_train)


best_param = lightgbm_search.best_params_
lgbm = LGBMRegressor(**best_param)


def mean_cross_val(model,X,y):
    score = cross_val_score(model,X,y,cv=5)
    mean = score.mean()
    return mean
#交叉验证分数

xgb.fit(X_train,y_train)
preds = xgb.predict(X_val)
preds_test_xgb = xgb.predict(test)
mae_xgb = mean_absolute_error(y_val,preds)
rmse_xgb = np.sqrt(mean_squared_error(y_val,preds))
score_xgb = xgb.score(X_val,y_val)
cv_xgb = mean_cross_val(xgb, x, y)

lgbm.fit(X_train, y_train)   
preds = lgbm.predict(X_val) 
preds_test_lgbm = lgbm.predict(test)
mae_lgbm = mean_absolute_error(y_val, preds)
rmse_lgbm = np.sqrt(mean_squared_error(y_val, preds))
score_lgbm = lgbm.score(X_val, y_val)
cv_lgbm = mean_cross_val(lgbm, x, y)
#结果分数

model_performances = pd.DataFrame({
    'Model':['XGBoost',"LGBM"],
    'CV(5)':[str(cv_xgb)[0:5], str(cv_lgbm)[0:5]],
    "MAE":[str(mae_xgb)[0:5], str(mae_lgbm)[0:5]],
    "RMSE":[str(rmse_xgb)[0:5], str(rmse_lgbm)[0:5]],
    'Score':[str(score_xgb)[0:5], str(score_lgbm)[0:5]]
}) 

print("Sorted by Score:")
print(model_performances.sort_values(by='Score',ascending=False))
#研究模型优劣并准备融合

def blend_models_predict(X, b, c):
        return ((b* xgb.predict(X)) + (c * lgbm.predict(X)))
subm = np.exp(blend_models_predict(test, 0.6, 0.4))
submission = pd.DataFrame({'Id': test.index,
                           'SalePrice': subm})

submission.to_csv("../../kaggle/working/submission.csv", index=False)
#准备提交
