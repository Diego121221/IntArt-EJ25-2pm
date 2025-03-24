import pandas as pd 
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt 
import seaborn as sns 


dataframe = pd.read_csv(r"C:\Users\Diego Covarrubias\Desktop\dat\IA\usuarios_win_mac_lin.csv")


X = np.array(dataframe.drop(['clase'], axis=1))  
y = np.array(dataframe['clase']) 
print(X.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = linear_model.LogisticRegression() 
model.fit(X_train, y_train)


predictions = model.predict(X_test) 


print(predictions[:5])

print(model.score(X, y))
