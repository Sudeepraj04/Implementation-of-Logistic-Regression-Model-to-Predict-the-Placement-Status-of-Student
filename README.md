## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results. 

## Program:
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: c r sudeep raj
RegisterNumber:24013567  
*/
data = pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)
data.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])```
``
## Output:
## HEAD
![ML 5 1](https://github.com/user-attachments/assets/5442c421-0d4f-4c05-b488-3de34602d503)
## COPY
![ML 5 2 COPY](https://github.com/user-attachments/assets/d0fe9423-12df-43e5-93cf-9aad530486d9)
## FIT TRANSFORM
![ML 5 3 FIT TRANSFORM](https://github.com/user-attachments/assets/3e6881b5-9862-47c7-b14c-00d418eb2c21)
## LOGISTIC REGRESSION
![ML 5 4 LOGISTIC REGRESION](https://github.com/user-attachments/assets/e4e11b82-3239-4c90-90c3-6bdcbd21d00d)
## ACCURACY SCORE
![ML 5 5 ACCURACY](https://github.com/user-attachments/assets/da69f076-1530-4124-9258-009d4a3ba01b)
## CONFUSION MATRIX
![ML 5 6 CONFUSSION Mtrix](https://github.com/user-attachments/assets/63b2465e-f5d3-4bc7-b528-812a78416744)
## CLASSIFICATION REPORT
![ML 5 7 CLASSIFICATION REPORT](https://github.com/user-attachments/assets/20e13e37-bc20-4ad5-b70a-4ae137bc90e9)
## PREDICITION
![ML 5 8 PREDICTION](https://github.com/user-attachments/assets/cf5c029b-2247-4e0c-8802-fdf351782337)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
