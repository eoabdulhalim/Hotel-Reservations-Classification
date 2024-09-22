# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 15:37:07 2024

@author: Essam Omar
"""
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import math
from Model_Final import BookingModel, encoding_Categorical

app = Flask(__name__)
booking_model = pickle.load(open('modelfinal.pkl','rb'))

@app.route('/')
def home():
    return render_template('hotelclass.html')

@app.route('/predict', methods=['POST'])
def predict():

    column_names = ['no_of_adults','no_of_children', 'no_of_weekend_nights','no_of_week_nights'
           , 'type_of_meal_plan','lead_time','arrival_year','arrival_month'
           ,'arrival_date', 'market_segment_type','repeated_guest',
           'avg_price_per_room', 'no_of_special_requests']  # replace with actual names


    form_values = request.form.to_dict(flat=True)


    form_values['avg_price_per_room'] = float(form_values['avg_price_per_room'])

    # Convert form values to DataFrame
    input_data = pd.DataFrame([form_values], columns=column_names)

    # Make prediction
    prediction = booking_model.Predict_Booking_Status(input_data)
    if prediction[0] == 1:
        output = "'Cancelled'"
        status_class = "cancelled"
    else:
        output = "'Not Cancelled'"
        status_class = "not-cancelled"

    return render_template('hotelclass.html', prediction_text=f"The Booking Status of This Reservation is Expected to be {output}", status_class=status_class)

if __name__ == '__main__':
    app.run()