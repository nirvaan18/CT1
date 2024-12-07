import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load the trained model
wholesalemodel = load('wholesalemodel.pkl')

def predict_channel(region, fresh, milk, grocery, frozen, detergents_paper, delicassen):
    """Function to predict the channel (Horeca or Retail)"""
    # Map region input to numerical values
    region_mapping = {'Lisbon': 1, 'Oporto': 2, 'Other': 3}
    region_numeric = region_mapping[region]

    # Create input DataFrame
    inputs_dict = {
        'Region': [region_numeric],
        'Fresh': [fresh],
        'Milk': [milk],
        'Grocery': [grocery],
        'Frozen': [frozen],
        'Detergents_Paper': [detergents_paper],
        'Delicassen': [delicassen]
    }
    input_df = pd.DataFrame(inputs_dict)

    # Predict channel
    channel = wholesalemodel.predict(input_df)[0]
    return channel

def app_layout():
    """Function to define the app layout"""
    st.title('Wholesale Customer Channel Prediction')
    st.header('Enter the details:')

    # Region input with no default selection
    region = st.radio('Region:', ['Lisbon', 'Oporto', 'Other'], index=None)  # Remove default selection

    # User prompt
    st.markdown("**Enter or adjust the spend values using the +/- buttons:**")
    
    # Numerical inputs with specific min and max values
    fresh = st.number_input('Fresh (Annual Spend Range min 3 - max 112151):', value=10)
    milk = st.number_input('Milk (Annual Spend Range min 55 - max 73498 ):',value=10)
    grocery = st.number_input('Grocery (Annual Spend Range min 3 - max 92780):',value=10)
    frozen = st.number_input('Frozen (Annual Spend Range min 25 - max 60869):',value=10)
    detergents_paper = st.number_input('Detergents_Paper (Annual Spend Range min 3 - max 40827):',value=10)
    delicassen = st.number_input('Delicassen (Annual Spend Range min 3 - max 47943):',value=10)

    # Create a placeholder for displaying the prediction
    prediction_placeholder = st.empty()

    # Check if the region is selected and the button is enabled only if region is selected
    button_disabled = not region

    # Prediction button
    if st.button('Predict Channel', disabled=button_disabled):
        # If region is not selected, show a warning
        if not region:
            st.warning("Please select a region to enable the prediction model.")  # Show warning if no region is selected
        else:
            channel = predict_channel(region, fresh, milk, grocery, frozen, detergents_paper, delicassen)
            channel_name = 'Horeca' if channel == 1 else 'Retail'
            
            # Display the predicted channel in the output field using markdown for styling
            prediction_placeholder.markdown(
                f"<div style='font-size:20px; font-weight:bold; color:blue;'>"
                f"Predicted Channel: {channel_name} (Channel {channel})</div>",
                unsafe_allow_html=True,
            )

if __name__ == '__main__':
    app_layout()
