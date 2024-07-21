# -*- coding: utf-8 -*-

#导入包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#线性回归1

##模型准备
my_model = LinearRegression()

##这节课随便造了两串数据
train_data = np.array([[1,1], [1,2], [2,2], [2,3]])   #生成一个用于train的数据集，4个样本（4个中括号），每个样本俩特征（每个中括号里俩数）
train_label = np.dot(train_data, np.array([1,2])) + 3.14 + 2*np.random.rand(len(train_data))   #四个样本对应的标签

##训练模型
my_model.fit(train_data, train_label)

##测试模型

test_data = np.array([[2,1], [3,3]])
test_label = np.dot(test_data, np.array([1,2])) + 3.14 + 2*np.random.rand(len(test_data))

#这里就是开始对测试集进行测试
R_2 = my_model.score(test_data, test_label)

#测试返回结果为R2，我们就打印出来
print(R_2)


#这里我们对模型进行解释
weight = my_model.coef_


#这里将解释结果打印
print(weight)


#这里是假如我们想要模型对数据进行预测并打印预测结果
pre_label = my_model.predict(test_data)
print(pre_label)




#逻辑回归 Logistic Regression

#导入库
import numpy as np
from sklearn.linear_model import LogisticRegression   #导入模型所在的类
from sklearn.model_selection import train_test_split   #这个类用于划分数据集


#加载数据
from sklearn.datasets import load_breast_cancer   #导入数据集的包
cancer_data = load_breast_cancer()
data = cancer_data['data']   #569个样本，每个样本30个维度，即30个特征
label = cancer_data['target']   #569个label


#划分数据集
train_data2, test_data2, train_label2, test_label2 = train_test_split(data,
                                                                  label, 
                                                                  test_size=0.2,    #随机划分20%数据给测试集
                                                                  random_state=42)  #是否保证每次划分的结果一致

#模型初始化
also_my_model = LogisticRegression(random_state=42,   #再设置一次保证结果重复性
                                   penalty='l2',   #选择l2（默认）的正则化方式
                                   C=1)   #这个数字越小，正则化效果越强


#拟合模型
also_my_model.fit(train_data2, train_label2)


## 再训练集上评估模型
ACC_past = also_my_model.score(train_data2, train_label2)
print(ACC_past)


## 评估测试模型
ACC_pre = also_my_model.score(test_data2, test_label2)
print(ACC_pre)



## 只进行分类预测，不计算准确率
pre_classifying = also_my_model.predict(test_data2)
print(pre_classifying)


## 查看预测概率
probability = also_my_model.predict_proba(test_data2)
print(probability)


## 解释模型
weight2 = also_my_model.coef_   #注意，coef是属性，后面不跟括号，跟_
print(weight2)


## 绘制ROC曲线

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


y_pre = probability[:,1]   #分类为1的概率
fpr, tpr, thresholds = roc_curve(test_label2, y_pre)   #先计算fpr、tpr
roc_auc = auc(fpr, tpr)   #在计算AUC
plt.plot(fpr, tpr,'r--', label = 'ROC(area = {0:.2f})'.format(roc_auc), lw = 2)   #绘制ROC曲线
plt.xlim([-0.05,1.05])   #设置x轴上下限
plt.ylim([-0.05,1.05])   #设置y轴上下限
plt.xlabel('False Positive Rate')   #x轴标签
plt.ylabel('True Positive Rate')   #y轴标签
plt.title('ROC Curve')
plt.legend(loc = "Lower Right")
plt.show()








#验证集和调参
import numpy as np
from sklearn.linear_model import LogisticRegression   #导入模型所在的类
from sklearn.model_selection import train_test_split   #这个类用于划分数据集


from sklearn.datasets import load_breast_cancer   #导入数据集的包
cancer_data = load_breast_cancer()
data = cancer_data['data']   #569个样本，每个样本30个维度，即30个特征
label = cancer_data['target']   #569个label


#这里和之前一样，先把数据集划分为训练集和测试集两部分
train_data, test_x, train_label, test_y = train_test_split(data, label, test_size = 0.1, random_state = 42)


#之后就是不一样的地方了，我们接下来把训练集进一步划分成训练集和验证集
train_x, val_x, train_y, val_y = train_test_split(train_data, train_label, test_size = 0.2, random_state = 42)


#初始化和拟合
model_too = LogisticRegression(random_state = 42, penalty = 'l2', C = 0.5)
model_too.fit(train_x, train_y)


#分别评估验证集和训练集的模型
ACC_train = model_too.score(train_x, train_y)
ACC_val = model_too.score(val_x, val_y)
print(ACC_train)
print(ACC_val)

#得到结果后我们就根据结果调整之前代码里的参数，直到我们无法再提高验证集评估效果
#然后我们将调参后的模型去测试

ACC_test = model_too.score(test_x, test_y)
print(ACC_test)


#同样的假如我们只是想预测测试集的分类和概率：
test_classifying = model_too.predict(test_x)
test_probability = model_too.predict_proba(test_x)
print(test_classifying)
print(test_probability)


#另外，我们也不要忘记解释模型的权重
weight233 = model_too.coef_
bias233 = model_too.intercept_
print(weight233)
print(bias233)



# 画图
#导入需要的包
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


#开始画图
y_pre = test_probability[:,1]   #分类为1的概率
fpr, tpr, thresholds = roc_curve(test_y, y_pre)   #先计算fpr、tpr
roc_auc = auc(fpr, tpr)   #在计算AUC
plt.plot(fpr, tpr,'r--', label = 'ROC(area = {0:.2f})'.format(roc_auc), lw = 2)   #绘制ROC曲线
plt.xlim([-0.05,1.05])   #设置x轴上下限
plt.ylim([-0.05,1.05])   #设置y轴上下限
plt.xlabel('False Positive Rate')   #x轴标签
plt.ylabel('True Positive Rate')   #y轴标签
plt.title('ROC Curve')
plt.legend(loc = "lower right")
plt.show()
