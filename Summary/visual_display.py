numeric_ = X.select_dtypes(exclude=['object']).drop(['MSSubClass'],axis=1).copy()
numeric_.columns

#分类为数字类型，'MSSubClass'是建筑类型，分为object

cat_train = X.select_dtypes(include=['object']).copy()
cat_train['MSSubClass'] = X['MSSubClass']
cat_train.columns

#分为字符类型，加上'MSSubClass'

disc_num_var = ['OverallQual','OverallCond','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
                'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']
cont_nume_var = []

#分类离散变量与连续变量

for i in numeric_.columns:
    if i not in disc_num_var:
        cont_nume_var.append(i)

fig = plt.figure(figsize=(18,16))
for index,col in enumerate(cont_nume_var):
    plt.subplot(6,4,index+1)
    sns.distplot(numeric_.loc[:,col].dropna(),kde=False)
fig.tight_layout(pad=1.0)

#绘制条形图，研究数据分布


fig = plt.figure(figsize=(18,16))
for index,col in enumerate(cont_nume_var):
    plt.subplot(6,4,index+1)
    sns.boxplot(y=col,data=numeric_.dropna())
fig.tight_layout(pad=1.0)
#绘制箱型图，研究异常值

fig = plt.figure(figsize=(18,16))
for index,col in enumerate(disc_num_var):
    plt.subplot(6,4,index+1)
    sns.countplot(x=col,data=numeric_.dropna())
fig.tight_layout(pad=1.0)
#研究离散数据的分布


fig = plt.figure(figsize=(18,20))
for index in range(len(cat_train.columns)):
    plt.subplot(9,5,index+1)
    sns.countplot(x=cat_train.iloc[:,index],data=cat_train.dropna())
    plt.xticks(rotation=90)
fig.tight_layout(pad=1.0)
#研究字符数据的分布

plt.figure(figsize=(14,12))
correlation = numeric_.corr()
sns.heatmap(correlation,mask = correlation < 0.8,linewidth=0.5)
#对多重共线性分析
#多重共线性导致模型无法准确判断一个变量对结果的具体贡献，为了保持模型的稳定性，需删除。


numeric_train = train.select_dtypes(exclude=['object'])
correlation = numeric_train.corr()
correlation[['SalePrice']].sort_values(['SalePrice'],ascending=False)
#研究变量与预测之间的相关性


fig = plt.figure(figsize=(20,20))
for index in range(len(numeric_train.columns)):
    plt.subplot(10,5,index+1)
    sns.scatterplot(x=numeric_train.iloc[:,index],y='SalePrice',data=numeric_train.dropna())
#绘制散点图，研究关系


plt.figure(figsize=(20,10))
plt.title('number of missing rows')
missing_count = pd.DataFrame(X.isnull().sum(),columns=['sum']).sort_values(by=['sum'],ascending=False).head(20).reset_index()
missing_count.columns=['features','sum']
sns.barplot(x='features',y='sum',data=missing_count)
#研究缺失值


fig,axes = plt.subplots(1,2,figsize=(10,5))
sns.regplot(x=numeric_train['MoSold'],y='SalePrice',data=numeric_train,ax=axes[0],line_kws={'color':'black'})
sns.regplot(x=numeric_train['YrSold'], y='SalePrice', data=numeric_train, ax = axes[1],line_kws={'color':'black'})
fig.tight_layout(pad=2.0)
#研究无意义值


out_col = ['LotFrontage','LotArea','BsmtFinSF1','TotalBsmtSF','GrLivArea']
fig = plt.figure(figsize=(20,5))
for index,col in enumerate(out_col):
    plt.subplot(1,5,index+1)
    sns.boxplot(y=col,data=X)
fig.tight_layout(pad=1.5)
#研究异常值


pd.DataFrame(X.isnull().sum(),columns=['sum']).sort_values(by=['sum'],ascending=False).head(30)
#准备处理缺失值

neigh_lot = X.groupby('Neighborhood')['LotFrontage'].mean().reset_index(name='LotFrontage_mean')
neigh_garage = X.groupby('Neighborhood')['GarageArea'].mean().reset_index(name='GarageArea_mean')

fig,axes = plt.subplots(1,2,figsize=(20,8))
axes[0].tick_params(axis='x',rotation=90)
sns.barplot(x='Neighborhood',y='LotFrontage_mean',data=neigh_lot,ax=axes[0])
axes[1].tick_params(axis='x', rotation=90)
sns.barplot(x='Neighborhood', y='GarageArea_mean', data=neigh_garage, ax=axes[1])
#以neiborhood为单位研究变量的均值分布

plt.figure(figsize=(10,6))
plt.title("Before transformation of SalePrice")
dist = sns.distplot(train['SalePrice'],norm_hist=False)
#对总体SalePrice的分布研究，发现是右偏分布

plt.figure(figsize=(10,6))
plt.title("After transformation of SalePrice")
dist = sns.distplot(np.log(train['SalePrice']),norm_hist=False)
#变成正态分布，模型总是认为数据是正态分布的，确保稳定性

model_performances = pd.DataFrame({
    'Model':['XGBoost',"LGBM"],
    'CV(5)':[str(cv_xgb)[0:5], str(cv_lgbm)[0:5]],
    "MAE":[str(mae_xgb)[0:5], str(mae_lgbm)[0:5]],
    "RMSE":[str(rmse_xgb)[0:5], str(rmse_lgbm)[0:5]],
    'Score':[str(score_xgb)[0:5], str(score_lgbm)[0:5]]
}) 

print("Sorted by Score:")
print(model_performances.sort_values(by='Score',ascending=False))
#对不同模型分数对比，确定Blending的分数


