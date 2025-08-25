import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
# Title
st.title("Car Attributes Form")

# Create a form
with st.form("car_form"):
    symboling = st.number_input("Symboling", value=0, min_value=-2, max_value=5)
    aspiration = st.selectbox("Aspiration", ["std", "turbo"])
    carbody = st.selectbox("Car Body Type", ["sedan", "hatchback", "wagon", "hardtop", "convertible"])
    drivewheels = st.selectbox("Drive Wheels", ["fwd", "rwd", "4wd"])
    enginelocation = st.selectbox("Engine Location", ["front", "rear"])
    enginetype = st.selectbox("Engine Type", ["ohc", "ohcf", "ohcv", "dohc", "l", "rotor", "dohcv"])
    cylindernumber = st.selectbox("Cylinder Number", ["four", "six", "five", "eight", "two", "twelve", "three"])
    fuelsystem = st.selectbox("Fuel System", ["mpfi", "2bbl", "idi", "1bbl", "spdi", "4bbl", "mfi", "spfi"])
    brand = st.selectbox("Brand", [
        "toyota", "nissan", "mazda", "mitsubishi", "honda", "subaru", "volkswagen", "volvo",
        "peugeot", "dodge", "buick", "bmw", "audi", "plymouth", "saab", "porsche", "isuzu",
        "alfa-romero", "chevrolet", "jaguar", "renault", "mercury"
    ])

    wheelbase = st.number_input("Wheelbase (inches)", value=86.0, min_value=60.0, max_value=150.0)
    carwidth = st.number_input("Car Width (inches)", value=60.0, min_value=50.0, max_value=80.0)
    carheight = st.number_input("Car Height (inches)", value=50.0, min_value=40.0, max_value=70.0)
    enginesize = st.number_input("Engine Size (cc)", value=70, min_value=50, max_value=6000)
    boreratio = st.number_input("Bore Ratio", value=2.5, min_value=2.0, max_value=5.0, format="%.2f")
    stroke = st.number_input("Stroke", value=2.5, min_value=2.0, max_value=6.0, format="%.2f")
    compressionratio = st.number_input("Compression Ratio", value=8.0, min_value=5.0, max_value=15.0, format="%.2f")
    horsepower = st.number_input("Horsepower", value=50, min_value=30, max_value=1000)
    peakrpm = st.number_input("Peak RPM", value=4000, min_value=1000, max_value=10000)
    citympg = st.number_input("City MPG", value=15, min_value=5, max_value=100)

    # Submit button
    submitted = st.form_submit_button("Submit")

    if submitted:
        st.success("Form submitted successfully!")

        # Load the model
        model_path = os.path.join("models", "model.pkl")
        model = pickle.load(open(model_path, "rb"))

        # Create dataframe for the model
        car_data = pd.DataFrame([{
            "symboling": symboling,
            "aspiration": aspiration,
            "carbody": carbody,
            "drivewheels": drivewheels,
            "enginelocation": enginelocation,
            "enginetype": enginetype,
            "cylindernumber": cylindernumber,
            "fuelsystem": fuelsystem,
            "brand": brand,
            "wheelbase": wheelbase,
            "carwidth": carwidth,
            "carheight": carheight,
            "enginesize": enginesize,
            "boreratio": boreratio,
            "stroke": stroke,
            "compressionratio": compressionratio,
            "horsepower": horsepower,
            "peakrpm": peakrpm,
            "citympg": citympg
        }])

        # Predict
        log_prediction = model.predict(car_data)[0]
        prediction = np.expm1(log_prediction)

        st.write(f"Predicted Value: {prediction}")
