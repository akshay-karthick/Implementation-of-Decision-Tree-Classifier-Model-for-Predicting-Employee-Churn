# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: AKSHAY KARTHICK ASR

RegisterNumber:  212224230015

```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:

## HEAD() AND INFO():
<img width="1434" height="698" alt="Screenshot 2025-10-27 142914" src="https://github.com/user-attachments/assets/8c7d1cd2-5c3d-45d0-bbc3-08ca8750c0fd" />

## NULL & COUNT:
<img width="1439" height="425" alt="Screenshot 2025-10-27 142935" src="https://github.com/user-attachments/assets/8e7ed5e7-ae1b-45c6-a224-ed4fa3f604de" />

<img width="1437" height="765" alt="Screenshot 2025-10-27 143005" src="https://github.com/user-attachments/assets/7192bbd3-2b94-425f-b6b5-1411230e4751" />


## ACCURACY SCORE:
<img width="669" height="168" alt="Screenshot 2025-10-27 143029" src="https://github.com/user-attachments/assets/e5fc9808-e045-4ae6-93a2-f404b03ed9fe" />


## DECISION TREE CLASSIFIER MODEL:
<img width="1447" height="580" alt="Screenshot 2025-10-27 143050" src="https://github.com/user-attachments/assets/b371a286-04f8-49fb-a286-5623e00d6bb9" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
