import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

def run_time_series_forecast(df, date_col, target_col, periods=30):
    """
    Runs a time-series forecast using Prophet.
    Returns a Plotly Figure and a dataframe with the forecast.
    """
    try:
        from prophet import Prophet
    except ImportError:
        return None, "Prophet library is not installed. Please try forecasting with another method."
        
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
    
    # Ensure 'ds' is datetime
    try:
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)
    except Exception as e:
        return None, f"Could not convert the selected column '{date_col}' to a valid Date/Time format. Please select a column that only contains dates."
        
    # Ensure 'y' is numeric (coerces errors to NaN)
    prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
    
    # Replace infinities with NaN to drop them
    prophet_df['y'] = prophet_df['y'].replace([np.inf, -np.inf], np.nan)
    
    # Drop NaNs
    prophet_df = prophet_df.dropna(subset=['ds', 'y'])
    
    # Sort by date and reset index
    prophet_df = prophet_df.sort_values(by='ds').reset_index(drop=True)
    
    if prophet_df.empty or len(prophet_df) < 10:
        return None, "Not enough valid data points for forecasting."
        
    if prophet_df['y'].nunique() <= 1:
        return None, "Target column must have variance (more than one unique value) for forecasting."
        
    if prophet_df['ds'].nunique() <= 1:
        return None, "Date column must have more than one unique date for forecasting."
        
    try:
        # Initialize and fit Prophet model
        m = Prophet(daily_seasonality=False)
        m.fit(prophet_df)
        
        # Create future dataframe and predict
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        
        # Create an interactive Plotly chart
        fig = go.Figure()
        
        # Actual data
        fig.add_trace(go.Scatter(
            x=prophet_df['ds'], y=prophet_df['y'],
            mode='markers', name='Actual',
            marker=dict(color='black', size=4)
        ))
        
        # Predicted line
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines', name='Forecast',
            line=dict(color='blue')
        ))
        
        # Uncertainty intervals
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'],
            mode='lines', line=dict(width=0),
            showlegend=False, name='Upper Bound'
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'],
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)',
            showlegend=False, name='Lower Bound'
        ))
        
        fig.update_layout(
            title=f"Forecast of {target_col}",
            xaxis_title="Date",
            yaxis_title=target_col,
            hovermode='x unified'
        )
        
        return fig, forecast
        
    except Exception as e:
        return None, f"Error running Prophet forecast: {str(e)}"
        
def run_regression_prediction(df, target_col, feature_cols):
    """
    Runs a simple Random Forest Regression.
    Returns feature importances and a scatter plot of Actual vs Predicted.
    """
    # Ensure columns exist and have no NaNs
    valid_cols = [col for col in feature_cols if col in df.columns] + [target_col]
    temp_df = df[valid_cols].dropna()
    
    X = temp_df[valid_cols[:-1]]  # exclude target
    y = temp_df[target_col]
    
    if len(X) < 10 or X.empty:
        return None, None, "Not enough data points after removing NaNs."
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    
    # Calculate error
    rmse = root_mean_squared_error(y_test, preds)
    
    # Plot Actual vs Predicted
    fig = px.scatter(
        x=y_test, y=preds, 
        labels={'x': 'Actual Value', 'y': 'Predicted Value'},
        title=f"Actual vs Predicted {target_col} (RMSE: {rmse:.2f})"
    )
    # Add identity line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    fig.add_shape(
        type="line", line=dict(dash="dash"),
        x0=min_val, y0=min_val, x1=max_val, y1=max_val
    )
    
    # Feature importances
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return fig, importances, "Success"
