---
title: Titanic Problem from Kaggel
date: 2018-7-23 11:44
tags: [kaggle]
---

## 1. 问题##

泰坦尼克号与冰山相撞，2224名船员及乘客中，逾1500人丧生。本题是一个二分类问题，需要我们根据题目提供的部分游客信息和存活情况来预测什么样的人有可能在这次灾难中存活。

<!-- more -->

> Competition Description
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

## 2. 数据 ##

注意下面对数据的处理都只写了在train或者test里面的处理，其实应该对train和test做一样的特征工程，后面我会封装好函数让它们过一样的流程。

### 2.1. 初识数据 ###

kaggle提供了2组数据：train.csv和test.csv，可以直接打开Excel看数据的样子，也可读入数据后print出来

~~~ python
import pandas as pd
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print train.head(6)
print train.describe()
~~~

![Alt text](/img/kaggle/titanic/Figure1.png)
图 1

由数据可知提供的信息包含有PassengerId(乘客ID)，Survived(获救与否)，Pclass(舱位等级)，Name(姓名)，Sex(性别)，Age(年龄)，SibSp(船上兄弟姐妹数目)，Parch(船上兄弟姐妹数目)，Ticket(票号)，Fare(票价)，Cabin(船舱号)和Embarked(登船港口)。其中Survived代表的就是Y值，我们也知道看到数据是有缺失的，为了更好了解数据的完整度，我们可以用两种方法查看。


~~~ python
print train.info()
~~~

{% img [class names] /img/kaggle/titanic/Figure2.png 300 500 %}
图 2


~~~ python
print test.insull.sum()
~~~

{% img [class names] /img/kaggle/titanic/Figure3.png 150 350 %}
图 3

由图2我们可以知道train文档共有891组数据，其中Age，Cabin，Embarked这三个特征值有不同程度的数据缺失，且10项信息的数据类型也是一致的。图三告诉我们test中也有缺失信息

### 2.2. 缺失数据 ###

缺失值处理一般采用两种方法：直接删除，数据补全。其中数据补全又分特殊值填充、平均值填充、众数填充、插值填充等。

#### 2.2.1. Age特征数据缺失处理 ####

由图2可知Age数据类型是float64，缺失177个。数据缺失将近2成，直接删除会导致本来小的数据值更小不太可取，而且test里面Age数据也有缺失不可能删除。于是我这里选择了数据补全，在补全之前我们也可以画图看看train中Age特征的数据分布,也可调describe()函数

~~~ python
from matplotlib import pyplot as plt
age = train['Age'].dropna() ### 删除缺失数据
plt.xlabel('Age')
plt.ylabel('Num')
plt.hist(age,20) 
plt.show()
~~~

{% img [class names] /img/kaggle/titanic/Figure4.png 400 250 %}

~~~ python
print train['Age'].describe()
~~~

{% img [class names] /img/kaggle/titanic/Figure5.png 200 500 %}

平均值是29，大多数人是20-40岁这个区间，因为缺失数有2成之多，假如统一补充平均值，可能会引起模型最后不小的偏差，这里我选择的了填补特殊值-10，年龄数据本身是不会有负数的，这里采用负数就是将其缺失作为一种特征值进行使用

~~~ python
train['Age'].fillna(value = -10,inplace = True)
~~~


#### 2.2.2. Cabin特征数据缺失处理 ####

由图2可知Cabin船舱号的数据类型是object,缺失8成左右的数据，不可直接删除缺失改数据的样本，也不可能补充众数或平均值。这里有三个选择：填充特殊值表示船舱号为空，以有无船舱号作为新的特征，或者直接删除Cabin特征不参与模型学习。因为船舱号大部分数据缺失，在刚开始跑baseline时，可以直接不用这个特征，我这里选择以有无船舱号作为新的特征。

~~~ python
train['Cabin'].fillna(value = 'UN',inplace = True)
train['have_Cabin'] =  train['Cabin'].apply(lambda x: 0 if x == 'UN' else 1)
train = train.drop('Cabin', axis=1)
~~~


#### 2.2.3. Embarked数据缺失处理 ####

Embarked代表乘客的登船港口，数据类型是object，train.csv只缺失2个且test.csv中无缺失，故这里可采取直接删除缺失该数据的样本或者填补众数的方法，对结果影响都不大。下面是采用填补众数的方法。

~~~ python
print train['Embarked'].describe()
~~~

{% img [class names] /img/kaggle/titanic/Figure6.png 225 280 %}

由上可知有3个不同的登船港口，其中S港口人最多，故S为众数

~~~ python
train['Embarked'].fillna(value = 'S',inplace = True)
~~~

现在train中所有所有缺失数据都已经处理完毕，我们可以再检查数据的完整度

~~~ python
print train.info()
~~~

{% img [class names] /img/kaggle/titanic/Figure2.png 300 500 %}

### 2.2.4. Fare数据缺失处理 ####

Fare缺失的数据只在test集里面存在，且只有两个，我这里采用了插值法处理。

~~~ python
test['Fare'] = test['Fare'].interpolate()
~~~

## 3. 特征与结果关联 ##

每个乘客都有自己的姓名，Ticket，和PassengerIdID，虽然不同的姓氏可能有代表不同的权利等级或者隐藏亲属关系，但船舱和价位可以得出些信息且数据本身有给出乘客在船上亲朋好友的个数，故姓名和PassengerId这两个信息我暂且没有用上。

### 3.1. Pclass ###

~~~ python
print set(train['Pclass'])
print train['Survived'].corr(train['Pclass']) ### 相关性

set([1, 2, 3])
-0.338481035961
~~~

由上可知分1，2，3等座，且和Survived相关性是负相关(-0.34)，即一等座存活率高些，紧接是二等座。我们还可以画图看看具体的数目情况。

~~~ python
dead = train[train.Survived==0].groupby('Pclass')['Survived'].count()
alive = train[train.Survived==1].groupby('Pclass')['Survived'].count()
ax = plt.figure(figsize=(8,4)).add_subplot(111)
ax.bar([1,2,3], dead, color='b', alpha=0.6, label='dead')
ax.bar([1,2,3], alive, color='g', bottom=dead, alpha=0.6, label='alive')
ax.legend(fontsize=16, loc='best')
ax.set_xticks([1,2,3])
ax.set_xticklabels(['Pclass1', 'Pclass2', 'Pclass3'], size=15)
plt.ylabel('Num', size = 15)
ax.set_title('Pclass & Surveved', size=20)
plt.show()
~~~

{% img [class names] /img/kaggle/titanic/Figure8.png 400 600 %}

由图可更直观看出三等座的存活率远远比不上其它，这说明这个特征对分类还是很有用处的。

### 3.2. Sex ###

#### 3.2.1. Sex & Survived ####

我们同样可以观察下获救与否和性别的相关性，以及不同性别获救的人数直方图

~~~ python
train['is_male'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
train = train.drop('Sex', axis=1)
print train['Survived'].corr(train['is_male'])


0.543351380658
~~~

~~~ python
dead = train[train.Survived==0].groupby('is_male')['Survived'].count()
alive = train[train.Survived==1].groupby('is_male')['Survived'].count()
ax = plt.figure(figsize=(8,4)).add_subplot(111)
ax.bar([1, 2], dead, color='b', alpha=0.6, label='dead')
ax.bar([1, 2], alive, color='g', bottom=dead, alpha=0.6, label='alive')
ax.legend(fontsize=16, loc='best')
ax.set_xticks([1,2])
ax.set_xticklabels(['Female', 'male'], size=15)
plt.ylabel('Num', size = 15)
ax.set_title('Sex & Surveved', size=20)
plt.show()
~~~

{% img [class names] /img/kaggle/titanic/Figure9.png 400 600 %}

由上可知相关性为0.54，女性的获救几率比男性大得多。

#### 3.2.2. Pclass, Sex & Survived ####

下图中绿色代表获救代表，可以看出在一二等座的女性获救几率是很高的。

~~~ python
ax = plt.figure(figsize=(10,4)).add_subplot(111)
sns.violinplot(x='is_male', y='Pclass', hue='Survived', data=train, split=True)
ax.set_xlabel('Sex',size=20)
ax.set_ylabel('Pclass',size=20)
ax.set_xticklabels(['Male','Female'], size=18)
ax.legend(fontsize=25,loc='best')
ax.set_title('Sex, Pclass & Surveved', size=20)
plt.show()
~~~

{% img [class names] /img/kaggle/titanic/Figure10.png 600 800 %}

### 3.3. Age ###

#### 3.3.1. Age & Survived ####

前面年龄缺失数据已补全，接下来我们看看Age数据和获救与否的关系图, 根据图像来确定分组年龄界限(这里取了大约交点的位置来分组),在不同的年龄段获救几率不尽相同。

~~~ python
fig, ax = plt.subplots(figsize=(10,5))
ax.set_title('Age & Survived')
k1 = sns.distplot(train[train.Survived==0].Age,  hist = False, color='b', ax=ax, label='dead')
k2 = sns.distplot(train[train.Survived==1].Age, hist = False, color='y', ax=ax, label='alive')
ax.set_xlabel('Age')
ax.legend(fontsize=16)
plt.show()
~~~

{% img [class names] /img/kaggle/titanic/Figure11.png 500 650 %}

~~~ python
def age_map(x):
    if x < 0:
        return 1
    elif x < 16 and x > 0:
        return 2
    elif x < 27 and x > 15:
        return 3
    elif x < 58 and x > 26:
        return 4
    else:
        return 5          
train['Age_map'] = train['Age'].apply(lambda x: age_map(x))
train = train.drop('Age', axis=1)
dead = train[train.Survived==0].groupby('Age_map')['Survived'].count()
alive = train[train.Survived==1].groupby('Age_map')['Survived'].count()
ax = plt.figure(figsize=(8,4)).add_subplot(111)
ax.bar([1,2,3,4,5], dead, color='b', alpha=0.6, label='dead')
ax.bar([1,2,3,4,5], alive, color='g', bottom=dead, alpha=0.6, label='alive')
ax.legend(fontsize=16, loc='best')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(['Age1', 'Age2', 'Age3', 'Age4', 'Age5'], size=15)
plt.ylabel('Num', size = 15)
ax.set_title('Pclass & Surveved', size=20)
plt.show()
~~~

{% img [class names] /img/kaggle/titanic/Figure12.png 400 600 %}

### 3.4. SibSp & Parch ###

家属和朋友都是亲近的人所以这里我把他们都放在一起。

~~~ python
train['family&friend'] = train['SibSp'] + train['Parch']
train = train.drop(['SibSp', 'Parch'], axis=1)
train['is_with_f'] = train['family&friend'].apply(lambda x: 0 if x == 0 else 1)
print train['Survived'].corr(train['family&friend'])
print train['Survived'].corr(train['is_with_f'])

0.0166389892827
0.2033670857
~~~

可以看到Survived和乘客是否有亲朋好友的相关度比和亲朋好友的个数要更大所以我保留了train['is_with_f']，删除了train['family&friend']，当然也许有朋友会想保留两个特征，这都没问题。

~~~ python
train = train.drop(['family&friend'], axis=1)
~~~


### 3.5. 特征相关性 ###

先把Embarked变为one_hot编码的。然后我们看看最后数据长什么样。

~~~ python
train = train.get_dummies(train, prefix=['is'])
train.head(5)
~~~

![Alt text](/img/kaggle/titanic/Figure13.png)

最后我们可以总得看看特征之间的相关程度，我们可以直接调用函数查看数值，当然看热图的话会更直观。

~~~ python
import matplotlib
matplotlib.use('TkAgg')

train_corr = train.corr()
sns.set()
f, ax = plt.subplots()
sns.heatmap(train_corr,  ax = ax, annot = True, linewidths=1, square=True, cmap="Blues")
ax.set_xticklabels(train_corr.index, size=10)
ax.set_yticklabels(train_corr.columns[::-1], size=10)
ax.set_title('all train feature corr', fontsize=20)
plt.show()
~~~

{% img [class names] /img/kaggle/titanic/Figure14.png 500 500 %}

## 4. 模型 ##

模型的选择有很多，我这里先选择了随机森林。下面是我的源代码，后期还会再调参数以及用
ensemble。

~~~ python
# coding = utf-8
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

def age_map(x):
    if x < 0:
        return 1
    elif x < 16 and x > 0:
        return 2
    elif x < 27 and x > 15:
        return 3
    elif x < 58 and x > 26:
        return 4
    else:
        return 5 
    
def clean_data(data):
    data = data.drop('PassengerId', axis=1) 
    data = data.drop('Name', axis=1)
    data = data.drop('Ticket', axis=1)
     
    data['Age'].fillna(value = -10,inplace = True)
    data['Age_map'] = data['Age'].apply(lambda x: age_map(x))
    data = data.drop('Age', axis=1)
    
    data['Cabin'].fillna(value = 'UN',inplace = True)
    data['have_Cabin'] =  data['Cabin'].apply(lambda x: 0 if x == 'UN' else 1)
    data = data.drop('Cabin', axis=1)   

    data['is_male'] = data['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    data = data.drop('Sex', axis=1)
    

    data['family&friend'] = data['SibSp'] + data['Parch']
    data = data.drop(['SibSp', 'Parch'], axis=1)
    data['is_with_f'] = data['family&friend'].apply(lambda x: 0 if x == 0 else 1)       
    data = data.drop(['family&friend'], axis=1)
    
    data['Fare'] = data['Fare'].interpolate()
    data['Embarked'].fillna(value = 'S',inplace = True)
    data = pd.get_dummies(data, prefix=['is'])

    return data

def save_predict_result(ID, result):
    submission = pd.DataFrame({"PassengerId": ID["PassengerId"],
        "Survived": result })
    submission.to_csv('submission.csv', index = False)
    print 'save_result succ!!!'
    return


def choose_parameter(model, X, Y):
    para = { 'n_estimators': [10,100,300,800], 
             'max_features': ['log2', 'sqrt','auto'], 
             'criterion': ['entropy', 'gini'],
             'max_depth': [2，6, 9], 
             'min_samples_split': [5，10，20, 30],
             'min_samples_leaf': [10, 20]
            }
    scoring = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(model, para, scoring = scoring, n_jobs = 4, verbose = 1)
    grid_obj = grid_obj.fit(X, Y.values.ravel())
    best_model= grid_obj.best_estimator_
    print best_model
    return best_model
    
def main():
    train = pd.read_csv('train.csv')
    test= pd.read_csv('test.csv')
    test_Y_ID = pd.read_csv('gender_submission.csv')
    
    train_X, train_Y = clean_data(train).drop(['Survived'], axis=1), train['Survived']
    test_X, test_Y= clean_data(test), test_Y_ID['Survived']

    model = RandomForestClassifier()
    model = choose_parameter(model, train_X, train_Y)
    model.fit(train_X, train_Y)
    predict_Y = model.predict(test_X)
    
    train_score = model.score(train_X, train_Y)
    test_score = model.score(test_X, test_Y)
    
    print train_score, test_score
    save_predict_result(test_Y_ID, predict_Y)
    return

if __name__ == '__main__':
    main()
~~~
