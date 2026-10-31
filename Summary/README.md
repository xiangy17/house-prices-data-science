# 1. 数据加载与初步观察

读取数据：导入所需的库，读取 train.csv 和 test.csv。

合并数据：将训练集（去掉标签 SalePrice）和测试集上下合并（concat），以便统一进行特征工程，防止数据泄漏并保证特征维度一致。

# 2. 探索性数据分析

特征分类：将特征分为数值型（连续/离散）和类别型。

分布与异常值：绘制直方图（Distplot）、箱线图（Boxplot），查看数据分布及明显离群点。

相关性分析：使用热力图（Heatmap）查看特征间的多重共线性，并计算各特征与目标变量 SalePrice 的相关系数，找出关键特征（如 OverallQual, GrLivArea）。

# 3. 数据清洗与特征工程

剔除异常值：手动删除了如 LotArea, GrLivArea 等特征中明显过大的极端离群点。

删除无用特征：删除了缺失率极高、或单一值占比 >96% 的高度倾斜特征（如 PoolQC, Street）。

缺失值填补：
  
  类别特征：填补 'NA'，或按 Neighborhood 分组用众数（mode）填补。
  
  数值特征：按 Neighborhood 分组用均值（mean）填补。

特征编码：将有序的类别（如品质：Ex, Gd）映射为数值；将部分特征二值化（0/1）；最后使用 pd.get_dummies 进行独热编码。

构造新特征：创造了总面积（TotalSF）、总卫浴数（TotalBath）、总门廊面积（TotalPorch）等组合特征。

# 4. 数据预处理


目标变量变换：对 SalePrice 进行对数变换（np.log），使其符合正态分布，利于模型学习。

特征缩放：使用 RobustScaler 对数值特征进行标准化处理（对异常值鲁棒）。

数据集划分：将训练集切分为 X_train 和 X_val（8:2 划分）。

# 5. 模型训练与调优


模型选择：选用了两大集成树模型：XGBoost 和 LightGBM。

超参数调优：使用 RandomizedSearchCV（随机搜索结合 5 折交叉验证）寻找最优超参数。

# 6. 模型评估与结果输出

模型评估：在验证集上计算 MAE、RMSE 和 R² Score 对比模型表现（LGBM 略优）。

模型融合 (Blending)：将 XGBoost 和 LightGBM 的预测结果按 0.6 : 0.4 的权重进行加权平均。

反变换与提交：使用 np.exp() 将预测结果还原为真实价格，生成 submission.csv 提交文件。
