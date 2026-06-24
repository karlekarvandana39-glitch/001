import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(
    page_title="Stock Forecast ARIMA",
    layout="wide"
)

st.title("📈 Stock Price Forecasting using ARIMA")



# User Input

ticker = st.text_input(
    "Enter Stock Symbol",
    "RELIANCE.NS"
)


forecast_month = st.date_input(
    "Forecast Till",
    pd.to_datetime("2027-06-01")
)



# -------------------------
# Download Data
# -------------------------

if st.button("Run Forecast"):


    st.subheader("Fetching Data...")


    data = yf.download(
        ticker,
        period="5y",
        interval="1d"
    )


    if data.empty:
        st.error("Invalid Stock Symbol")
        st.stop()


    df = data[['Close']].dropna()



    # -------------------------
    # Historical Chart
    # -------------------------

    st.subheader(
        "Last 5 Years Stock Price"
    )


    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        df.index,
        df.Close,
        label="Closing Price"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid()


    st.pyplot(fig)



    # -------------------------
    # ARIMA MODEL
    # -------------------------

    st.subheader(
        "ARIMA Forecast"
    )


    model = ARIMA(
        df['Close'],
        order=(5,1,0)
    )


    model_fit = model.fit()



    future_dates = pd.date_range(
        start=df.index[-1],
        end=forecast_month,
        freq="B"
    )


    forecast = model_fit.forecast(
        steps=len(future_dates)
    )


    forecast_df = pd.DataFrame(
        {
            "Date":future_dates,
            "Forecast":forecast.values
        }
    )


    forecast_df.set_index(
        "Date",
        inplace=True
    )



    # Forecast Chart

    fig2, ax2 = plt.subplots(figsize=(12,5))


    ax2.plot(
        df.index,
        df.Close,
        label="Actual"
    )


    ax2.plot(
        forecast_df.index,
        forecast_df.Forecast,
        label="Forecast"
    )


    ax2.set_title(
        f"{ticker} Forecast"
    )

    ax2.legend()
    ax2.grid()


    st.pyplot(fig2)



    # -------------------------
    # June 2027 Prediction
    # -------------------------

    result = forecast_df[
        forecast_df.index.strftime("%Y-%m")
        ==
        "2027-06"
    ]


    st.subheader(
        "June 2027 Forecast Price"
    )


    st.dataframe(result)



    st.success(
        "Forecast Completed Successfully"
    )
