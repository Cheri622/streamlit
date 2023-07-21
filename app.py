#執行: streamlit run app.py
import streamlit as st
import numpy as np
 
st.title('City House Price prediction')
 
st.write('---')
 
# area of the house
area = st.slider('Area of the house?', 1000, 4000, 1500)
 
# no. of bedrooms in the house
bedroom = st.number_input('No. of bedrooms?', min_value=0, step=1)
 
# no. of balconies in the house
balcony = st.radio('No. of balconies?', (0, 1, 2 , 3))
 
# No. of bathrooms? 
bath = st.number_input('No. of bathrooms?', min_value=0, step=1)
 
import joblib
 
lr = joblib.load('lr_model.sav')

if st.button('Predict House Price'):
    cost = lr.predict(np.array([[bedroom, area, bath, balcony]]))
    st.text(cost[0])
    
#While passing the data to the predict function, we need to pass it as a 2-d array. Hence, we have converted the input data to a NumPy array. Also, the predict function returns a 1-d array, so while printing the value we have written cost[0] to get the sole value in the returned array.    