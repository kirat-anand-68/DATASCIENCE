import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

from heart2 import prediction

df=pd.read_csv('/Users/Kirat/OneDrive/Desktop/creditcard.csv')
legit=df[df['Class']==0]
fraud=df[df['Class']==1]

legit_sample=legit.sample(n=len(fraud), random_state=2)
data=pd.concat([legit_sample, fraud],axis=0)
X=df.drop("Class",axis=1)
y=df['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2,stratify=y)

model=LogisticRegression()
model.fit(X_train,y_train)

train_acc= accuracy_score(model.predict(X_train),y_train)
test_acc= accuracy_score(model.predict(X_test),y_test)

st.title("CREDIT CARD FRAUD DETECTION")
input_df=st.text_input("ENTER ALL REQUIRED FEATURES VALUES")

input_df_splitted=input_df.split(',')

submit=st.button("SUBMIT")

if submit:
    features=np.asarray(input_df_splitted,dtype=np.float64)
    prediction=model.predict(features.reshape(1,-1))

    if prediction[0]==0:
        st.write("Failed to predict")
    else:
        st.write("Fraud Detection")
