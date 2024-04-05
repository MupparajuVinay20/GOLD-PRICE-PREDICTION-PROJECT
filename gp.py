import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()



# Load the dataset from a pickle file
lrmodel = pd.read_pickle('linear_regression_model.pkl')
gp = pd.read_csv('Gold_data.csv',index_col=0, parse_dates=True)

gp1 = gp.resample('2M').mean()
m_gp= gp1.copy()

scaler.fit(gp1['price'].values.reshape(-1, 1))

# Transform the gold price data using the fitted scaler
gp1['price'] = scaler.transform(gp1['price'].values.reshape(-1, 1))
gp1['price'] = gp1['price'].diff()
gp.dropna(inplace=True)



# Define the sidebar navigation
nav = st.sidebar.radio("Navigation", ["Home", "Data", "Visualization", "Prediction"])

if nav == "Home":
    st.title("Home")
    st.write("""
    This App Predicts The Future Gold Prices.
    """)
    st.image("GLD.jpg",width=500)




if nav == "Data":
    st.title("Data")
    # Load and display the dataset
    gp = pd.read_csv('Gold_data.csv')
    st.write(gp)



if nav == "Visualization":
    gp = pd.read_csv('Gold_data.csv',index_col=0, parse_dates=True)

    gpm = gp.resample('2M').mean()
    scaler.fit(gpm['price'].values.reshape(-1, 1))

# Transform the gold price data using the fitted scaler
    gpm['price'] = scaler.transform(gpm['price'].values.reshape(-1, 1))
    # m_gp= gp1.copy()
    gpw = gp.resample('W').mean()
    gpy = gp.resample('Y').mean()
    gp_year_avg = gpy.groupby('date')['price'].mean().reset_index()

    st.title("Visualization")
    graph = st.selectbox("Select Preferred Graph ",["Days","Months","Weeks","Years","Avg gold price per year","linear model pred"])
    if graph == "Days":  
        st.write("## Gold Price Over Time (Days)")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the gold price
        ax.plot(gp.index, gp['price'], label='Gold Price')

        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Gold Price Over Time')

        # Add legend
        ax.legend()

        st.pyplot(fig)

    elif graph == "Months":  
        st.write("## Gold Price Over Time (Months)")
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot the prices of gold over time
        # plt.figure(figsize=(12, 6))
        ax.plot(gpm.index, gpm['price'], label='Gold Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Gold Price Over Time')
        ax.legend()
        st.pyplot(fig)

    elif graph == "Weeks":  
        st.write("## Gold Price Over Time (Weeks)")
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot the prices of gold over time
        # plt.figure(figsize=(12, 6))
        ax.plot(gpw['price'], label='Gold Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Gold Price Over Time')
        ax.legend()
        st.pyplot(fig)

    elif graph == "Years":  
        st.write("## Gold Price Over Time (Years)")
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot the prices of gold over time
        # plt.figure(figsize=(12, 6))
        ax.plot(gpy['price'], label='Gold Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Gold Price Over Time')
        ax.legend()
        st.pyplot(fig)

    elif graph == "Avg gold price per year":  
        st.write("## Gold Price Over Time (Years)")
        fig, ax = plt.subplots(figsize=(12, 6))
        # plt.figure(figsize=(12, 6))
        ax.pie(gp_year_avg['price'], labels=gp_year_avg['date'], autopct="%1.1f%%")
        ax.set_title('Average Gold Price per Year')
        st.pyplot(fig)

    elif graph == "linear model pred": 
        st.write("## Gold Price Over Time (Years)")
        fig, ax = plt.subplots(figsize=(12, 6))

                
        # Prepare the data
        X = gpm.index.to_numpy().reshape(-1, 1)
        y = gpm['price'].to_numpy()

        # Train the model
        lrmodel = LinearRegression()
        lrmodel.fit(X, y)
        X = X.astype(np.float64)

        # Make predictions
        lr_predictions = lrmodel.predict(X) 
        # plt.figure(figsize=(10,5))
        ax.plot(lr_predictions)
        ax.set_xlabel("Time")
        ax.set_ylabel("Gold Price")
        ax.set_title("Linear Regression Model Predictions")
        st.pyplot(fig)


if nav == "Prediction":
    
    st.title("Prediction")
    next_days = pd.date_range(gp1.index[-1], periods=12, freq='M')
    next_days_array = next_days.to_numpy().reshape(-1, 1)
    next_days_array = next_days_array.astype(float)
    next_days_predictions = lrmodel.predict(next_days_array)
    next_days_predictions = pd.DataFrame({'Date': next_days, 'forecasted Price': next_days_predictions}, index=range(len(next_days)))
    next_days_predictions['forecasted Price'] = next_days_predictions['forecasted Price'].values.reshape(-1, 1)
    denormalized_prices = scaler.inverse_transform(next_days_predictions['forecasted Price'].values.reshape(-1, 1)).flatten()
    denormalized_prices = pd.DataFrame(denormalized_prices, columns=['forecasted price'], index=next_days_predictions['Date'])
    denormalized_prices.index = denormalized_prices.index.strftime('%Y-%m-%d')  # Convert index to string

    selected_date = st.selectbox("Select Date for Forecast", denormalized_prices.index, index=len(denormalized_prices)-1)
    selected_price = denormalized_prices.loc[selected_date, 'forecasted price']

    st.write(f"Forecasted Gold Price for {selected_date} :   {selected_price:.2f}/-")


    st.write("Forecasted Gold Prices")
    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot the predicted gold price
    ax.plot(next_days_predictions['Date'], next_days_predictions['forecasted Price'], label='Predicted Gold Price')

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Predicted Gold Price ')

    # Add legend
    ax.legend()

    # Show the plot
    st.pyplot(fig)




    
