import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



df=pd.read_csv("Student.csv",index_col=0)


le=LabelEncoder()
df["Internet"]=le.fit_transform(df["Internet"])  # YES = 1, NO = 0 
df["Passed"]=le.fit_transform(df["Passed"])



Features=["StudyHours","Attendence","PassScore","SleepHours"]
scaler=StandardScaler()
df_scaled=df.copy()
df_scaled[Features]=scaler.fit_transform(df[Features])



X=df_scaled[Features] # Features
y=df_scaled["Passed"] # Target

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)


print("Classification Matrics")
print(classification_report(y_test, y_pred))
conf_matrix=(confusion_matrix(y_test,y_pred))
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues",xticklabels=["Fail","Pass"],yticklabels=["Fail","Pass"])
plt.xlabel("Predictor")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()


print("-------Predict Your Result--------")

try:
    study_hours=(float(input("Enter Study Hours:")))
    attendence=(float(input("Enter Attendence:")))
    passscore=(float(input("Enter past score:")))
    sleephours=(float(input("Enter sleep hours:")))


    user_input_df=pd.DataFrame({
      "StudyHours": [study_hours],
       "Attendence":[attendence],
       "PassScore":[passscore],
       "SleepHours":[sleephours]

    })

    user_input_scaled=scaler.transform(user_input_df)
    prediction=model.predict(user_input_scaled)[0]

    if prediction==1:
        print("pass")
    else:
        print("fail")

except Exception as e:
    print("An error occured:",e)        
    
plt.show()
    
    