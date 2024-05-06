import streamlit as st
import pickle
import numpy as np

# Load the trained model
load_model = pickle.load(open('C:/Users/Ayush Tushar Vadalia/Documents/Finance project(Streamlit)/trained_model.sav', 'rb')) 


def finance_pred(input_data):
    # Convert input data to numpy array
    input_data = np.array(input_data).reshape(1, -1)
    # Predict turnover
    prediction = load_model.predict(input_data)
    return f'Predicted Turnover in Crores: {prediction[0]}'

def main():
    # giving a title 
    st.title('Finance turnover prediction')

    # getting the input data from the user
    open_price = st.number_input('Open')
    high_price = st.number_input('High')
    low_price = st.number_input('Low')
    close_price = st.number_input('Close')
    shares_traded = st.number_input('Shares_Traded')

    # Code for Prediction
    turnover_in_crs = ''
    
    # Creating a button for prediction
    if st.button('Predict the turnover'):
        # Check if all input fields are filled
        if open_price and high_price and low_price and close_price and shares_traded:
            # Call prediction function
            input_data = [open_price, high_price, low_price, close_price, shares_traded]
            turnover_in_crs = finance_pred(input_data)
        else:
            turnover_in_crs = "Please fill in all input fields."

    st.success(turnover_in_crs)

if __name__ == '__main__':
    main()
