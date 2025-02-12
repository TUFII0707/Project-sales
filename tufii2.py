import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import xml.etree.ElementTree as ET

st.set_page_config(page_title="Sales & Business Analytics", layout="wide")

# XML Parser Function
def parse_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    data = []
    for record in root.findall(".//record"):
        row = {child.tag: child.text for child in record}
        data.append(row)
    return pd.DataFrame(data)

# Exponential Smoothing Forecast
def forecast_exponential(series, forecast_period):
    try:
        model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=7)
        fit_model = model.fit()
        return fit_model.forecast(forecast_period)
    except:
        model = ExponentialSmoothing(series, trend='add', seasonal=None)
        fit_model = model.fit()
        return fit_model.forecast(forecast_period)

# Linear Regression Forecast
def forecast_linear_regression(df, date_col, selected_col, forecast_period):
    df['Days'] = (df[date_col] - df[date_col].min()).dt.days
    X = df[['Days']]
    y = df[selected_col]
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(df['Days'].max() + 1, df['Days'].max() + forecast_period + 1).reshape(-1, 1)
    predictions = model.predict(future_days)
    future_dates = pd.date_range(df[date_col].max() + pd.Timedelta(days=1), periods=forecast_period)
    return pd.DataFrame({"Date": future_dates, "Linear Regression Forecast": predictions})

# Sidebar - Upload dataset
st.sidebar.header("Upload Sales Data")
file = st.sidebar.file_uploader("Upload CSV/XML file", type=["csv", "xml"])

if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xml"):
        df = parse_xml(file)

    df.columns = df.columns.str.strip()
    
    # Auto-detect Date column
    date_col = st.sidebar.selectbox("Select Date Column", options=df.columns)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col)

    # Auto-detect numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_col = st.sidebar.selectbox("Select Column for Forecasting", options=numeric_cols)

    # Handle missing values
    df[selected_col] = pd.to_numeric(df[selected_col], errors='coerce').ffill()

    st.title("📊 Sales & Business Analytics Dashboard")

    # Data Overview
    st.subheader("📌 Data Overview")
    st.write(df.head())
    st.write("Data Summary:")
    st.write(df.describe())

    # Forecasting Period
    forecast_period = st.sidebar.slider(
        "Select Forecasting Period (Days)", 
        7, 90, 30, 
        key="forecast_period_slider"  # Unique key
    )

    # Sales Forecasting
    st.subheader(f"📈 Forecasting {selected_col}")
    exp_forecast_df = pd.DataFrame({"Date": pd.date_range(df[date_col].max() + pd.Timedelta(days=1), periods=forecast_period),
                                    "Exponential Smoothing Forecast": forecast_exponential(df[selected_col], forecast_period)})

    linear_forecast_df = forecast_linear_regression(df, date_col, selected_col, forecast_period)

    # Plots
    fig_exp = px.line(exp_forecast_df, x="Date", y="Exponential Smoothing Forecast", title="Exponential Smoothing Forecast")
    fig_linear = px.line(linear_forecast_df, x="Date", y="Linear Regression Forecast", title="Linear Regression Forecast")

    st.plotly_chart(fig_exp)
    st.plotly_chart(fig_linear)

    # Economic Order Quantity (EOQ) Model
    st.subheader("📦 Economic Order Quantity (EOQ) Model")
    demand = st.number_input("Enter Annual Demand", min_value=1, value=1000)
    ordering_cost = st.number_input("Enter Ordering Cost per Order", min_value=1, value=50)
    holding_cost = st.number_input("Enter Holding Cost per Unit per Year", min_value=1, value=5)
    eoq = np.sqrt((2 * demand * ordering_cost) / holding_cost)
    st.metric("Optimal Order Quantity (EOQ)", f"{eoq:.2f}")

    # ABC Analysis
    st.subheader("🔠 ABC Analysis")
    if 'Product' in df.columns and selected_col in df.columns:
        df['Cumulative Revenue'] = df[selected_col].cumsum()
        df['Percentage'] = df['Cumulative Revenue'] / df[selected_col].sum() * 100
        df['Category'] = pd.cut(df['Percentage'], bins=[0, 70, 90, 100], labels=['A', 'B', 'C'])
        st.write(df[['Product', selected_col, 'Percentage', 'Category']])

    # Supply Chain Management Insights
    if 'Shipping Cost' in df.columns:
        st.subheader("🚛 Logistics & Shipping Analysis")
        fig_shipping = px.line(df, x=date_col, y='Shipping Cost', title="Shipping Cost Trends")
        st.plotly_chart(fig_shipping)

    # Automatically detect sales column
    sales_col = next((col for col in df.columns if "sales" in col.lower()), None)
    if sales_col is None:
        st.error("Could not detect a sales column. Please ensure your dataset has a column related to sales.")
    else:
        st.title("Sales Forecasting & Stock Analysis")

        # Key Metrics
        st.subheader("Key Business Metrics")
        total_sales = df[sales_col].sum()
        avg_sales = df[sales_col].mean()
        max_sales = df[sales_col].max()
        min_sales = df[sales_col].min()

        st.metric("Total Sales", f"{total_sales:,.2f}")
        st.metric("Average Sales", f"{avg_sales:,.2f}")
        st.metric("Highest Sales", f"{max_sales:,.2f}")
        st.metric("Lowest Sales", f"{min_sales:,.2f}")

        # Cost Analysis
        if 'Cost' in df.columns:
            st.subheader("Cost Analysis")
            total_cost = df['Cost'].sum()
            avg_cost = df['Cost'].mean()
            st.metric("Total Cost", f"{total_cost:,.2f}")
            st.metric("Average Cost", f"{avg_cost:,.2f}")

        # Profit Analysis
        if 'Cost' in df.columns and sales_col in df.columns:
            st.subheader("Profit Analysis")
            df['Profit'] = df[sales_col] - df['Cost']
            total_profit = df['Profit'].sum()
            avg_profit = df['Profit'].mean()
            st.metric("Total Profit", f"{total_profit:,.2f}")
            st.metric("Average Profit", f"{avg_profit:,.2f}")

        # Expensive Product Analysis
        if 'Product' in df.columns and 'Cost' in df.columns:
            st.subheader("Most Expensive Products")
            expensive_products = df.groupby('Product')['Cost'].mean().nlargest(5).reset_index()
            st.write("Top 5 Most Expensive Products:")
            st.write(expensive_products)

        # Sales Trends Visualization
        st.subheader("Sales Trends")
        fig = px.line(df, x=date_col, y=sales_col, title='Sales Over Time')
        st.plotly_chart(fig)

        # Stock Analysis
        stock_col = next((col for col in df.columns if "stock" in col.lower()), None)
        if stock_col:
            st.subheader("Stock Analysis")
            stock_threshold = st.slider(
                "Set Stock Threshold", 
                min_value=0, 
                max_value=int(df[stock_col].max()), 
                value=10, 
                key="stock_threshold_slider"  # Unique key
            )
            low_stock = df[df[stock_col] < stock_threshold]
            st.write(f"Products below stock threshold ({stock_threshold}):", low_stock)

        # Demand & Trends Analysis
        st.subheader("Demand & Trends")
        df['Month'] = df[date_col].dt.to_period('M').astype(str)
        monthly_sales = df.groupby('Month')[sales_col].sum().reset_index()
        fig2 = px.bar(monthly_sales, x='Month', y=sales_col, title='Monthly Sales Trends')
        st.plotly_chart(fig2)

        # Sales Forecasting
        st.subheader("Sales Forecasting")
        forecast_period_sales = st.slider(
            "Select Forecasting Period (Days)", 
            7, 90, 30, 
            key="sales_forecast_slider"  # Unique key
        )

        df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce').replace(0, np.nan).ffill()

        try:
            model = ExponentialSmoothing(df[sales_col], trend='add', seasonal='add', seasonal_periods=7)
            fit_model = model.fit()
            forecast = fit_model.forecast(forecast_period_sales)
        except:
            model = ExponentialSmoothing(df[sales_col], trend='add', seasonal=None)
            fit_model = model.fit()
            forecast = fit_model.forecast(forecast_period_sales)

        forecast_df = pd.DataFrame({"Date": pd.date_range(df[date_col].max() + pd.Timedelta(days=1), periods=forecast_period_sales), "Forecast": forecast})
        fig3 = px.line(forecast_df, x='Date', y='Forecast', title='Sales Forecast')
        st.plotly_chart(fig3)

        # Cost Forecasting
        if 'Cost' in df.columns:
            st.subheader("Cost Forecasting")
            forecast_period_cost = st.slider(
                "Select Forecasting Period (Days)", 
                7, 90, 30, 
                key="cost_forecast_slider"  # Unique key
            )

            df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce').replace(0, np.nan).ffill()

            try:
                cost_model = ExponentialSmoothing(df['Cost'], trend='add', seasonal='add', seasonal_periods=7)
                cost_fit_model = cost_model.fit()
                cost_forecast = cost_fit_model.forecast(forecast_period_cost)
            except:
                cost_model = ExponentialSmoothing(df['Cost'], trend='add', seasonal=None)
                cost_fit_model = cost_model.fit()
                cost_forecast = cost_fit_model.forecast(forecast_period_cost)

            cost_forecast_df = pd.DataFrame({"Date": pd.date_range(df[date_col].max() + pd.Timedelta(days=1), periods=forecast_period_cost), "Cost Forecast": cost_forecast})
            fig4 = px.line(cost_forecast_df, x='Date', y='Cost Forecast', title='Cost Forecast')
            st.plotly_chart(fig4)

        # Profit Forecasting
        if 'Profit' in df.columns:
            st.subheader("Profit Forecasting")
            forecast_period_profit = st.slider(
                "Select Forecasting Period (Days)", 
                7, 90, 30, 
                key="profit_forecast_slider"  # Unique key
            )

            df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce').replace(0, np.nan).ffill()

            try:
                profit_model = ExponentialSmoothing(df['Profit'], trend='add', seasonal='add', seasonal_periods=7)
                profit_fit_model = profit_model.fit()
                profit_forecast = profit_fit_model.forecast(forecast_period_profit)
            except:
                profit_model = ExponentialSmoothing(df['Profit'], trend='add', seasonal=None)
                profit_fit_model = profit_model.fit()
                profit_forecast = profit_fit_model.forecast(forecast_period_profit)

            profit_forecast_df = pd.DataFrame({"Date": pd.date_range(df[date_col].max() + pd.Timedelta(days=1), periods=forecast_period_profit), "Profit Forecast": profit_forecast})
            fig5 = px.line(profit_forecast_df, x='Date', y='Profit Forecast', title='Profit Forecast')
            st.plotly_chart(fig5)

        # Logistics Management Feature
        st.subheader("Logistics Management")

        if 'Shipping Cost' in df.columns:
            total_shipping_cost = df['Shipping Cost'].sum()
            avg_shipping_cost = df['Shipping Cost'].mean()
            st.metric("Total Shipping Cost", f"{total_shipping_cost:,.2f}")
            st.metric("Average Shipping Cost", f"{avg_shipping_cost:,.2f}")

            # Shipping Cost Visualization (Example)
            fig_shipping = px.line(df, x=date_col, y='Shipping Cost', title='Shipping Cost Over Time')
            st.plotly_chart(fig_shipping)

        if 'Delivery Time' in df.columns:
            avg_delivery_time = df['Delivery Time'].mean()  # Assuming Delivery Time is in a numerical format
            st.metric("Average Delivery Time", f"{avg_delivery_time:,.2f}")  # Customize unit as needed

            # Delivery Time Visualization (Example - Box Plot)
            fig_delivery = px.box(df, y='Delivery Time', title='Delivery Time Distribution')
            st.plotly_chart(fig_delivery)

        if 'Warehouse Location' in df.columns:
            st.write("Warehouse Locations:")
            warehouse_counts = df['Warehouse Location'].value_counts()
            st.bar_chart(warehouse_counts) # Or use plotly for more interactive map visualizations

else:
    st.write("📤 Upload a dataset to begin analysis.")