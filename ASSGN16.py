from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
%matplotlib inline
from sklearn.svm import SVC
from sklearn.datasets import load_wine
wine=load_wine()
x=(wine.data)
y=(wine.target)
x.shape,y.shape
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model=tree.DecisionTreeClassifier()
num_trees=100
model1=BaggingClassifier(base_estimator=model)
model1
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
model1.fit(x_train,y_train)
pred=model1.predict(x_test)
from sklearn import metrics
metrics.accuracy_score(y_test,pred)
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred)
from sklearn.ensemble import RandomForestClassifier
model2=RandomForestClassifier(random_state=1,max_depth=10 )
model2.fit(x_train,y_train)
pred1=model2.predict(x_test)
metrics.accuracy_score(y_test,pred1)
print(classification_report(y_test,pred1))
confusion_matrix(y_test,pred1)
from sklearn.ensemble import GradientBoostingClassifier
model3=GradientBoostingClassifier()
model3
model3.fit(x_train,y_train)
pred2=model3.predict(x_test)
metrics.accuracy_score(y_test,pred2)
confusion_matrix(y_test,pred2)
print(classification_report(y_test,pred2))
model4=LogisticRegression()
model5=tree.DecisionTreeClassifier()
model6=SVC()
estimators=[]
estimators.append(('logistic',model4))
estimators.append(('cart',model5))
estimators.append(('svm',model6))
from sklearn.ensemble import VotingClassifier
ensemble=VotingClassifier(estimators)
ensemble.fit(x_train,y_train)
pred4=ensemble.predict(x_test)
metrics.accuracy_score(y_test,pred4)
confusion_matrix(y_test,pred4)
print(classification_report(y_test,pred4))





