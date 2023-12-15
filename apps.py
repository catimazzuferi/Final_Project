import os
import streamlit as st
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np

# Obtener el directorio del script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo KNeighborsRegressor
with open(os.path.join(script_directory, 'FinalProject', 'knn_regressor_model2.pkl'), 'rb') as f:
    knn_regressor = pickle.load(f)

# Cargar el transformador MinMaxScaler
with open(os.path.join(script_directory, 'FinalProject', 'min_max_scaler2.pkl'), 'rb') as f:
    transformer = pickle.load(f)

# Cargar el encoder OneHotEncoder
with open(os.path.join(script_directory, 'FinalProject', 'one_hot_encoder2.pkl'), 'rb') as f:
    encoder = pickle.load(f)

# Cargar el conjunto de datos (dataframe) para extraer información
# sobre la dirección y el enlace de Street View
data = pd.read_csv(os.path.join(script_directory, 'FinalProject', 'databasic.csv'))

# Set background color and padding
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
            padding: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app
st.title('🏡 Welcome to the Price Prediction App')

# Button to navigate to Prediction section
if st.button("Predict Price"):
    st.session_state.prediction_page = True

# Button to view the dataset
if st.button("View Dataset"):
    st.title("Explore the Dataset:")
    st.table(data.head())

# Prediction page
if getattr(st.session_state, 'prediction_page', False):
    st.subheader('🔮 Prediction Menu')

    # User input for prediction
    st.sidebar.title('🔮 Prediction Menu')
    baths = st.sidebar.slider('Select baths:', min_value=1, max_value=10, value=3)
    beds = st.sidebar.slider('Select beds:', min_value=1, max_value=12, value=3)
    living_space = st.sidebar.slider('Select living_space (sqft):', min_value=2, max_value=16250, value=2000)

    # Dropdown for selecting state
    selected_state = st.sidebar.selectbox('Select state:', data['state'].unique())

    # Button to trigger predictions
    if st.sidebar.button("Make Predictions"):
        # Filter data based on user inputs
        filtered_data = data[
            (data['state'] == selected_state) &
            (data['beds'] == beds) &
            (data['baths'] == baths) &
            (data['living_space'] == living_space)
        ]

        # Make predictions
        input_features = np.array([beds, baths, living_space, selected_state]).reshape(1, -1)
        scaled_numerical_features = transformer.transform(input_features[:, :3])  # Scale numerical features
        encoded_input = encoder.transform(input_features[:, 3:]).toarray()  # One-hot encode categorical features
        final_input = np.concatenate([scaled_numerical_features, encoded_input], axis=1)
        predicted_price = knn_regressor.predict(final_input)[0]

        # Display predicted price
        st.sidebar.subheader(f'💰 Estimated price for {selected_state}:')
        st.sidebar.write(f'${float(predicted_price):,.2f}')

        # Calcular la diferencia absoluta entre el precio real y el precio predicho
        data['price_difference'] = np.abs(data['price'] - predicted_price)

        # Filtrar el DataFrame para mostrar solo las filas del mismo estado que el usuario seleccionado
        similar_listings = data[
            (data['state'] == selected_state) &
            (data['price_difference'] <= predicted_price[0])  # Asegurar que estamos comparando un valor único
        ].sort_values('price_difference').head(5)

        # Mostrar la información de las 5 filas con las diferencias más pequeñas
        st.subheader("🌟 Top 5 rows with nearest prices:")
        st.table(similar_listings[['state','baths','beds','living_space', 'address', 'price', 'price_difference', 'street_view_link']])

        # Eliminar la columna temporal creada
        data = data.drop('price_difference', axis=1)

        # Button to return to the main page
        if st.button("Back to Home"):
            st.session_state.prediction_page = False
