import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
data=pd.read_csv("/Users/Kirat/OneDrive/Desktop/Salary_Data.csv")
x=np.array(data["YearsExperience"]).reshape(-1,1)
lr=LinearRegression()
lr.fit(x,np.array(data["Salary"]))
st.markdown("<h1 style='text-align: center;'>Salary Prediction</h1>", unsafe_allow_html=True)

nav=st.sidebar.radio("Navigation",["Home","Prediction","Contribution"])
if nav== "Home":
    st.image('/Users/Kirat/OneDrive/Desktop/sal.jpg',width=800)
    if st.checkbox("Show Table"):
        st.table(data)

    graph=st.selectbox("What Kind of Graph?",["Non-Interactive","Interactive"])
    val= st.slider("filter-Data using years",0,20)
    data=data.loc[data["YearsExperience"]>=val]
    if graph == "Non-Interactive":
        fig=plt.figure(figsize=(10,5))
        plt.scatter(data["YearsExperience"],data["Salary"])
        plt.ylim(0)
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.tight_layout()
        st.pyplot(fig)
    if graph == "Interactive":
        layout=go.Layout(
            xaxis=dict(range=[0,16]),
            yaxis=dict(range=[0,2100000])
        )
        fig=go.Figure(data=go.Scatter(x=data["YearsExperience"],y=data["Salary"],mode="markers"),
                      layout=layout)
        st.plotly_chart(fig)


if nav=="Prediction":
    st.header("Know Your Salary")
    val=st.number_input("enter you exp",0.00,20.00,step=0.25)
    ex = np.array(val).reshape(1,-1)
    predict=lr.predict(ex)[0]

    if st.button("Predict"):
        st.success(f"Your Predicted Salary is **₹ {round(predict, 2)}**")

if nav == "Contribution":
    st.header("Contribute to our dataset")

    ex = st.number_input("Enter Your Experience", 0.0, 20.0, step=0.25)
    sal = st.number_input("Enter Your Salary", 0, 1000000, step=1000)

    if st.button("Submit"):
        to_add = pd.DataFrame(
            {"YearsExperience": [ex], "Salary": [sal]}
        )

        to_add.to_csv(
            "/Users/Kirat/OneDrive/Desktop/Salary_Data.csv",
            mode="a",
            header=False,
            index=False
        )

        st.success("✅ Data added successfully!")
