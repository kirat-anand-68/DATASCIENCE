import streamlit as st
#st.title("Super Simple Title")
#st.header("this is a header")
#st.subheader("Subheader")
#st.markdown("This is _MarkDown_")
#st.caption("Small text")
#code_example="""
#def greet(name):
#    print('hello',name)
#"""
#st.code(code_example,language="python")

#st.divider()
#st.image('C:/Users/Kirat/OneDrive/Desktop/kirat sign.jpg')'''
import pandas as pd

#st.title("Streamlit Elements Demo")
#st.subheader("DataFrame")
#df=pd.DataFrame({
  #  "Name":['Alice','Bob','Charlie','David'],
  #  "age":[25,32,37,45],
  #  "Occupation":['Engineering','Doctor','Artist','Chef']
#})

#st.dataframe(df)

#st.subheader("Data Editor")
#editable_df=st.data_editor(df)

#st.subheader("static Table")
#s#t.table(df)

#st.subheader("Metrics")
#st.metric(label="total Rows",value=len(df))
#s#t.metric(label="Average Age",value=round(df['age'].mean(),1))
#import matplotlib.pyplot as plt
import numpy as np
#st.title("Streamlit Charts Demo")

#chart_data=pd.DataFrame(
   # np.random.randn(20,3),
   # columns=['A','B','C']

#)

#Area Chart Section

#s#t.subheader("Area Chart")
#st.area_chart(chart_data)

#st.subheader("Bar Chart")
#st.bar_chart(chart_data)

#st.subheader("Line Chart")
#st.line_chart(chart_data)

#st.subheader("Scatter Chart")
#scatter_data=pd.DataFrame({
#'x':np.random.randn(100),
#'#y':np.random.randn(100)
#})

#st.scatter_chart(scatter_data)

#st.subheader("Map")

#map_data=pd.DataFrame(
  #  np.random.randn(100,2)/[50,50]+[30.1290, 77.2800],
  #  columns=['lat','lon']

#)
#st.map(map_data)

# import datetime
# import streamlit as st
#
# st.title("Streamlit Form Demo")
#
# form_values = {
#     "name": "",
#     "age": None,
#     "gender": "",
#     "Dob": None
# }
#
# with st.form(key="user_info_form"):
#     form_values["name"] = st.text_input("Enter Your Name:")
#     form_values["age"] = st.number_input("Enter Your Age:", min_value=1, max_value=120, step=1)
#     form_values["gender"] = st.selectbox("Gender", ["-- Select --", "Male", "Female"])
#     form_values["Dob"] = st.date_input(
#         "Enter Your Birthdate",
#         min_value=datetime.date(1900, 1, 1),
#         max_value=datetime.date.today()
#     )
#
#     submit_button = st.form_submit_button(label="Submit")
#
#     if submit_button:
#         # ✅ Custom Validation
#         if form_values["name"].strip() == "":
#             st.error("❌ Name cannot be empty")
#         elif form_values["gender"] == "-- Select --":
#             st.error("❌ Please select a gender")
#         elif form_values["Dob"] is None:
#             st.error("❌ Please select a valid birthdate")
#         else:
#             st.balloons()
#             st.success("✅ Form Submitted Successfully!")
#
#             st.write("### Entered Information:")
#             for key, value in form_values.items():
#                 st.write(f"**{key}:** {value}")
#
# counter =0
# st.write(f"Counter value: {counter}")
#
# if st.button("Increment Counter"):
#     counter +=1
#     st.write(f"Counter Incremented to{counter}")
# else:
#     st.write(f"Counter stays at{counter}")

## session state session state is something that we can use to store value within the same user session

if "counter" not in st.session_state:
    st.session_state.counter = 0
if st.button("Increment Counter"):
    st.session_state.counter +=1
    st.write(f"Counter incremented to {st.session_state.counter}")

if st.button("Reset"):
    st.session_state.counter=0
st.write(f"Counter value:{st.session_state.counter}")
