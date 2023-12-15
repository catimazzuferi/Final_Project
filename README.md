# Final_Project

## Overview

This repository contains a comprehensive dataset related to housing for the top 50 American cities by population. The data includes information on property prices, beds, baths, living space, and various demographic factors such as population density, median household income, and geographic coordinates.

## Data source

[DATA LINK](https://www.kaggle.com/datasets/jeremylarcher/american-house-prices-and-demographics-of-top-cities). 

### Files
- **databasic.csv:** Final data set.
- First model:
- **app.py:** Streamlit for the model that I coudnt get the prediction.
- **Final Project:** First model more precise but coudnt ask the input to the user.
- **min_max_scaler.pkl:** Scaler for the first model.
- **one_hot_encoder.pkl:** Encoder for the first model.
- **random_forest_regressor.pkl:** Regressor for first model.
- Second model (The one I present)
- **Final Project 2:** Second model the one that I work with in streamlit and presentation.
- **knn_regressor_model2.pkl:** Regressor for second model.
- **min_max_scaler2.pkl:** Scaler for the second model.
- **one_hot_encoder2.pkl:** Encoder for the first model.
- **apps.py:** Streamlit for the model of the presentation.

#### Data Fields

- **Zip Code:** Zip code within which the listing is present.
- **Price:** Listed price for the property.
- **Beds:** Number of beds mentioned in the listing.
- **Baths:** Number of baths mentioned in the listing.
- **Living Space:** The total size of the living space, in square feet, mentioned in the listing.
- **Address:** Street address of the listing.
- **City:** City name where the listing is located.
- **State:** State name where the listing is located.
- **Zip Code Population:** The estimated number of individuals within the zip code. Data from Simplemaps.com.
- **Zip Code Density:** The estimated number of individuals per square mile within the zip code. Data from Simplemaps.com.
- **County:** County where the listing is located.
- **Median Household Income:** Estimated median household income. Data from the U.S. Census Bureau.
- **Latitude:** Latitude of the zip code. Data from Simplemaps.com.
- **Longitude:** Longitude of the zip code. Data from Simplemaps.com.
- **Street View Link:** Link to Google Street View for the property.

### Data Analysis and Modeling

In this work, a predictive model was built to predict property prices based on the provided variables. 

### Tableau

Additionally, detailed analyses were conducted using Tableau to explore the relationship between price and different factors.
 **Tableau Analysis:** Utilized Tableau to create visualizations and conduct in-depth analyses of property prices in relation to various variables such as beds, baths, living space, and demographic factors.
Explore the interactive visualizations on Tableau:
- [Tableau Visualization ([link](https://public.tableau.com/app/profile/maria.catalina.mazzuferi/viz/FinalProject_17026402596450/Pricevs_LivingSpace?publish=yes))

### Web Scraping

Web scraping techniques were employed to gather links from Google Street View.

### Streamlit Prediction App

To facilitate user interaction and predictions, a Streamlit application was developed. The app allows users to input specific property features and receive predicted prices based on the built model.

## Presentation

Pdf: Presentationfinalproject
