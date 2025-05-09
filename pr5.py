import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv('Social_Network_Ads.csv') 
df
    
#input data 
x=df[['Age','EstimatedSalary']] 
 
#output data 
y=df['Purchased'] 


from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler() 
x_scaled = scaler.fit_transform(x) 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,random_state=0,test_size = 0.25)

    
x_train 
y_train 

from sklearn.linear_model import LogisticRegression 

 #creat the object 
classifier = LogisticRegression()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test) 
y_train.shape 
x_train.shape 
y_pred 
y_test

from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test,y_pred) 
y_test.value_counts() 

from sklearn.metrics import classification_report 
print(classification_report(y_test,y_pred)) 

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test, cmap='Blues')
plt.show()
