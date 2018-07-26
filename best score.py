import pandas as pd 
import numpy as np 
import re 
import sklearn 
import xgboost as xgb 
import seaborn as sns 
import matplotlib .pyplot as plt 
%matplotlib inline 


import plotly.offline as py 
py.init_notebook_mode(connected = True)
import plotly.graph_objs as go 
import plotly.tools as tls 

import warnings 
warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier,
    GradientBoostingClassifier,ExtraTreesClassifier)

from sklearn.svm import SVC 
from sklearn.cross_validation import KFold


#1.特征工程
train = pd.read_csv("D:/Postgraduate/kaggle/train.csv")
test = pd.read_csv("D:/Postgraduate/kaggle/test.csv")

PassengerId = test['PassengerId']

train.head(3)

full_data = [train,test]

#a.Pclass 
print(train[['Pclass','Survived']].groupby(['Pclass'],as_index = False).mean())

#b.Sex
print(train[['Sex','Survived']].groupby(['Sex'],as_index = False).mean())

#c.SibSp and Parch 
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print(train[['FamilySize','Survived']].groupby(['FamilySize'],as_index = False).mean())

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1,'IsAlone'] = 1
print(train[['IsAlone','Survived']].groupby(['IsAlone'],as_index = False).mean())

#4.Embarked
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
print (train[['Embarked','Survived']].groupby(['Embarked'],as_index = False).mean())

#5.Fare 
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'],4)

print(train[['CategoricalFare','Survived']].groupby(['CategoricalFare'],as_index = False).mean())


#6.Age
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()


    age_null_random_list = np.random.randint(age_avg - age_std,age_avg + age_std,size = age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)


train['CategoricalAge'] = pd.cut(train['Age'],5)

print(train[['CategoricalAge','Survived']].groupby(['CategoricalAge'],as_index = False).mean())

#7.Name 
def get_title(name):
    title_search = re.search('([A-Za-z]+)\.',name)
    #If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
print(pd.crosstab(train['Title'],train['Sex']))#交叉表

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col',\
        'Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

print(train[['Title','Survived']].groupby(['Title'],as_index = False).mean())

#mother 
for dataset in full_data:
    dataset['Mother'] = 0
    dataset.loc[(dataset['Sex'] == 'female') & (dataset['Parch'] > 0) & (dataset['Age'] > 18),'Mother'] = 1

print(train[['Mother','Survived']].groupby(['Mother'],as_index = False).mean())


#data cleaning 
for dataset in full_data:
    #dataset = dataset.dropna(axis = 0)
    #mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female':1,'male':0}).astype(int)
    

    #mapping titles
    title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


    #mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

    #mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    #mapping age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

#变量选择
drop_elements = ['PassengerId','Name','Ticket','Cabin','SibSp',\
                 'Parch','FamilySize']
data_train = train.drop(drop_elements, axis = 1)
data_train_1 = data_train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

data_test = test.drop(drop_elements, axis = 1)

data_test_1 = test.drop(drop_elements, axis = 1)

data_train_2 = data_train_1.values#变成array
data_test_2 = data_test.values

#Classifier Comparison 
import matplotlib.pyplot as plt 
import seaborn as sns #画图工具

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB #先验为高斯分布的朴素贝叶斯
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis 
from sklearn.linear_model import LogisticRegression
import xgboost as xgb 
from sklearn.neural_network import MLPClassifier

classifiers = [KNeighborsClassifier(3),
              SVC(probability = True),
              DecisionTreeClassifier(),
              RandomForestClassifier(),
              AdaBoostClassifier(),
              GradientBoostingClassifier(),
              GaussianNB(),
              LinearDiscriminantAnalysis(),
              QuadraticDiscriminantAnalysis(),
              LogisticRegression(),
              xgb.XGBClassifier(),
              ExtraTreesClassifier(),
              MLPClassifier()]

log_cols = ['Classifier','Accuracy']

log = pd.DataFrame(columns = log_cols)

sss = StratifiedShuffleSplit(n_splits = 10, test_size = 0.1,random_state = 0)

features = ['Pclass','Sex','Age','Fare','Embarked','IsAlone','Title']
#X = data_train_1[features]
#y = data_train_1[['Survived']]

X = data_train_2[0::, 1::]#array切片
y = data_train_2[0::, 0]
acc_dict = {}

for train_index, test_index in sss.split(X,y):
    X_train,X_test = X[train_index], X[test_index]
    y_train,y_test = y[train_index], y[test_index]

    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test,train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf,acc_dict[clf]]], columns = log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes('muted')
sns.barplot(x = 'Accuracy',y = 'Classifier',data = log,color = 'b')


#删除数据框
#del df

#Prediction  SVM
candidate_classifier = SVC()
candidate_classifier.fit(data_train_2[0::,1::],data_train_2[0::,0])
y_pred = candidate_classifier.predict(data_test)

result_svc = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(),'Survived':y_pred.astype(np.int32)})
#输出结果
result_svc.to_csv("D:/Postgraduate/kaggle/result_svc.csv",index=False)


result_svc_1 增加了Mother这个特征



#Prediction RF 0.74162
candidate_rf = RandomForestClassifier()
candidate_rf.fit(data_train_2[0::,1::],data_train_2[0::,0])
y_rf = candidate_rf.predict(data_test)

result_RF = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(),'Survived':y_rf.astype(np.int32)})

result_RF.to_csv("D:/Postgraduate/kaggle/result_RF.csv",index=False)

candidate_rf.fit(data_train_2[0::,1::],data_train_2[0::,0]).feature_importances_


#xgb
gbm = xgb.XGBClassifier(
    n_estimators = 2000,\
    max_depth = 4,\
    min_child_weight = 2,\
    gamma = 0.9,\
    subsample = 0.8,\
    colsample_bytree = 0.8,\
    objective = 'binary:logistic',\
    nthread = -1,\
    scale_pos_weight = 1
    ).fit(data_train_2[0::,1::],data_train_2[0::,0])

y_xgb = gbm.predict(data_test.values)

result_xgb = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(),'Survived':y_xgb.astype(np.int32)})

result_xgb.to_csv("D:/Postgraduate/kaggle/result_xgb.csv",index=False)



from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
#2018.07.09 选择最好的6个模型进行调参
#Hyperparameter tunning for best models 
### META MODELING WITH RandomForestClassifier,SVC,AdaBoostClassifier,GradientBoostingClassifier
#,ExtraTreesClassifier,XGBClassifier
#rfc Parameters tunning 
RFC = RandomForestClassifier()

##search grid for optimal parameters 
rf_param_grid = {'max_depth':[None],\
                'max_features':[1,3,10],\
                'min_samples_split':[2,3,10],\
                'min_samples_leaf':[1,3,10],\
                'bootstrap':[False],\
                'n_estimators':[100,300],\
                'criterion':['gini']}

gsRFC = GridSearchCV(RFC, param_grid = rf_param_grid, cv = sss, scoring = 'accuracy',n_jobs = 1,verbose = 1)

gsRFC.fit(X_train,y_train)

RFC_best = gsRFC.best_estimator_

#Best score
gsRFC.best_score_



#SVC Parameters tunning 
#这里不能用其他kernel因为会报错：ValueError: X should be a square kernel matrix
SVMC = SVC(probability = True)
svc_param_grid = {'kernel':['rbf'],\
                'gamma':[0.001, 0.01, 0.1, 1],\
                'C':[1,10,50,100,200,300,1000]}

gsSVMC = GridSearchCV(SVMC,\
                     param_grid = svc_param_grid,\
                     cv = sss,\
                    scoring = 'accuracy',\
                     n_jobs = 2,\
                     verbose = 1)

gsSVMC.fit(X_train,y_train)

SVMC_best = gsSVMC.best_estimator_

#Best Score 
gsSVMC.best_score_



#AdaBoost Parameters tunning 
#选择CART分类树作为基分类器
#DTC = DecisionTreeClassifier() base_estimator 默认是CART

ada = AdaBoostClassifier(random_state = 7)

ada_param_grid = {'algorithm':['SAMME','SAMME.R'],\
                'n_estimators':[10,30,50,100,200,300],\
                'learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}

gsadaDTC = GridSearchCV(ada,\
                        param_grid = ada_param_grid,\
                        cv = sss,\
                        scoring = 'accuracy',\
                        n_jobs = 2,\
                        verbose = 1)

gsadaDTC.fit(X_train,y_train)

ada_best = gsadaDTC.best_estimator_

gsadaDTC.best_score_





#Gradient Boosting Parameters tunning 
GBC = GradientBoostingClassifier()

gb_param_grid = {'loss':['deviance'],\
                'n_estimators':[100,200,300],\
                'learning_rate':[0.1,0.05,0.01],\
                'max_depth':[4,8],\
                'min_samples_leaf':[100,150],\
                'max_features':[0.3,0.1]
                }

gsGBC = GridSearchCV(GBC,\
                    param_grid = gb_param_grid,\
                    cv = sss,\
                    scoring = 'accuracy',\
                    n_jobs = 2,\
                    verbose = 1)

gsGBC.fit(X_train,y_train)

GBC_best = gsGBC.best_estimator_

gsGBC.best_score_





#Extra Trees parameter tunning 
ExtC = ExtraTreesClassifier()

ex_param_grid = {'max_depth':[None],\
                'max_features':[1,3,10],\
                'min_samples_split':[2,3,10],\
                'min_samples_leaf':[1,3,10],\
                'bootstrap':[False],\
                'n_estimators':[100,300],\
                'criterion':['gini','entropy']}


gsExtc = GridSearchCV(ExtC,\
                    param_grid = ex_param_grid,\
                    cv = sss,\
                    scoring = 'accuracy',\
                    n_jobs = 2,\
                    verbose = 1)

gsExtc.fit(X_train,y_train)

ExtC_best = gsExtc.best_estimator_

gsExtc.best_score_





#xgb parameter tunning 
xgb = XGBClassifier()

xgb_param_grid = {'learning_rate':[0.01,0.05,0.1,0.15,0.2],\
                 'max_depth':[3,4,5,6],\
                 'n_estimators':[10,50,100,200,300],\
                 'objective':['binary:logistic'],\
                 'gamma':[i/10.0 for i in range(0,5)],\
                 'min_child_weight':[6,8,10,12],\
                 'subsample':[i/10.0 for i in range(0,6)],\
                 'colsample_bytree':[i/10.0 for i in range(6,10)],\
                 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
                 }

gsxgb = GridSearchCV(xgb,\
                    param_grid = xgb_param_grid,\
                    cv = sss,\
                    scoring = 'accuracy',\
                    n_jobs = 2,\
                    verbose = 1)

gsxgb.fit(X_train,y_train)

xgb_best = gsxgb.best_estimator_

gsxgb.best_score_


xgb_best = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=0.4, learning_rate=0.05,
       max_delta_step=0, max_depth=4, min_child_weight=6, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0.01,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=0.4).fit(data_train_2[0::,1::],data_train_2[0::,0])

y_xgb_best = xgb_best.predict(data_test.values)

result_xgb_best = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(),'Survived':y_xgb_best.astype(np.int32)})

result_xgb_best.to_csv("D:/Postgraduate/kaggle/result_xgb_best.csv",index=False)


#画出以上留个模型的learning_curve
#用sklearn的learning_curve得到training_score和cv_score,使用matplotlib画出learning curve
def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,
                        n_jobs=1,train_sizes=np.linspace(.05,1.,20),verbose=0,plot=True):


    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    plt.fill_between(train_sizes,test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_scores_mean,'o-',color="r",
             label="Training score")
    plt.plot(train_sizes,test_scores_mean,'o-',color="g",
             label="Cross-validation score")
    plt.legend(loc="best")

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = ((train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1]))

    return plt, midpoint, diff


g = plot_learning_curve(gsRFC.best_estimator_,'RF meaning curves',X_train,y_train,cv = sss)
g = plot_learning_curve(gsSVMC.best_estimator_,'SVM meaning curves',X_train,y_train,cv = sss)
g = plot_learning_curve(gsadaDTC.best_estimator_,'RF meaning curves',X_train,y_train,cv = sss)
g = plot_learning_curve(gsGBC.best_estimator_,'RF meaning curves',X_train,y_train,cv = sss)
g = plot_learning_curve(gsExtc.best_estimator_,'RF meaning curves',X_train,y_train,cv = sss)
#g = plot_learning_curve(gsxgb.best_estimator_,'RF meaning curves',X_train,y_train,cv = sss)



#Feature importance of tree based classifiers 
#基于分类器计算特征的重要性
nrows = 3
ncols = 2
fig,axes = plt.subplots(nrows = nrows, ncols = ncols, sharex = 'all', figsize = (15,15))

names_classifiers = [('Random Forest',RFC_best),\
                    ('SVC',SVMC_best),\
                    ('AdaBoost',ada_best),\
                    ('Gradient Boost',GBC_best),\
                    ('Extra Trees',ExtC_best),\
                    ('xgboost',xgb_best)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
