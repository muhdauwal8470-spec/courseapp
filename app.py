#import packages
import pandas as pd
import joblib
import streamlit as st

#load model using joblib
model = joblib.load("course_model.pkl")
ct = joblib.load("course_encoder.pkl")

#real data
goal = st.text_input("Enter your goal: ")
hobby = st.text_input("Enter your hobby: ")

#predict now
if st.button("Submit"):
    sample = pd.DataFrame({
        "goal": [goal],
        "hobby": [hobby]
    })

    encoded_sample = ct.transform(sample)
    prediction = model.predict(encoded_sample) 

    st.success(f"Recommeded course: {prediction}")

