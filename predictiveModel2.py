import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from prophet import Prophet
import numpy as np
from tabulate import tabulate

# Load and preprocess data
def load_and_preprocess_data(file_path):
    pd.set_option('display.max_rows', 250)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    total_lines = sum(1 for _ in open(file_path))
    print("Total lines: ", total_lines - 73109)

    # Read the header row first to get the column names
    header_df = pd.read_csv(file_path, nrows=1)

    chunks = pd.read_csv(file_path, skiprows=range(2, 73110), chunksize=10000)
    df = pd.concat([chunk for chunk in tqdm(chunks, total=(total_lines - 73109) // 10000, desc="Reading CSV")])

    df.columns = header_df.columns

    # Melt df to make it long format with one row per country-year-indicator
    melted_df = df.melt(
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        var_name='Year',
        value_name='Value'
    )

    # group by 'Indicator Name' and count the number of non-null entries for each indicator
    indicator_counts = melted_df.groupby('Indicator Name')['Value'].count().reset_index(name='Count')

    # sort the indicators based on the count in descending order
    sorted_indicators = indicator_counts.sort_values(by='Count', ascending=False).reset_index(drop=True)

    # filter indicators with counts less than 7000
    filtered_indicators = sorted_indicators[sorted_indicators['Count'] >= 7000]
    relevant_indicators = filtered_indicators['Indicator Name'].tolist()

    filtered_df = melted_df[melted_df['Indicator Name'].isin(relevant_indicators)]

    return filtered_df

# Function to prepare data for a specific country and indicator
def prepare_data_for_country_indicator(filtered_df, country_name, target_indicator):
    country_data = filtered_df[filtered_df['Country Name'] == country_name]
    country_pivot = country_data.pivot_table(
        values='Value',
        index='Year',
        columns='Indicator Name',
        aggfunc='mean'
    )

    # Handle missing values
    country_pivot = country_pivot.fillna(method='ffill').fillna(method='bfill')
    country_pivot.reset_index(inplace=True)

    country_pivot['Year'] = pd.to_datetime(country_pivot['Year'].astype(str) + '-01-01')

    df_prophet = country_pivot[['Year', target_indicator]].rename(columns={'Year': 'ds', target_indicator: 'y'})
    
    return country_pivot, df_prophet

def generate_historical_prophet_predictions(df_prophet):
    model = Prophet()
    model.fit(df_prophet)
    historical_predictions = model.predict(df_prophet.drop('y', axis=1))
    return historical_predictions

# Generate future Prophet predictions
def generate_future_prophet_predictions(df_prophet, predict_year):
    model = Prophet()
    model.fit(df_prophet)
    max_year = df_prophet['ds'].dt.year.max()
    future = model.make_future_dataframe(periods=predict_year - max_year, freq='YE')
    future_predictions = model.predict(future)
    return future_predictions

def train_time_series_with_folds_autoreg_prophet_features(country_pivot, df_prophet, target_indicator, predict_year, lags=[1, 2, 3, 4, 5, 6, 7, 8]):
    # Create a dataframe with all the new features created with Prophet
    historical_prophet_features = generate_historical_prophet_predictions(df_prophet)
    
    df = pd.merge(country_pivot, historical_prophet_features[['ds', 'yhat']], left_on='Year', right_on='ds', how='inner')
    df.drop('ds', axis=1, inplace=True)
    df.set_index('Year', inplace=True)
    # print(tabulate(df.head(1), headers='keys', tablefmt='pqsl'))
    
    # Create some lag variables using Prophet predictions (yhat column)
    for lag in lags:
        df[f'yhat_lag_{lag}'] = df['yhat'].shift(lag)
    df.dropna(axis=0, how='any', inplace=True)

    print(tabulate(df.head(1), headers='keys', tablefmt='pqsl'))
    
    X = df.drop(target_indicator, axis=1)
    y = df[target_indicator]
    
    # Split the data into training (80%) and testing (20%) sets
    train_size = int(0.8 * len(df))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # # Define and train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Calculate MAE
    mae = np.round(mean_absolute_error(y_test, predictions), 3)
    print('MAE: ', mae)
    
    # Generate future lagged features for prediction
    future_prophet_features = generate_future_prophet_predictions(df_prophet, predict_year)
    future_df = pd.merge(country_pivot, future_prophet_features[['ds', 'yhat']], left_on='Year', right_on='ds', how='inner')
    future_df.drop('ds', axis=1, inplace=True)
    future_df.set_index('Year', inplace=True)

    for lag in lags:
        future_df[f'yhat_lag_{lag}'] = future_df['yhat'].shift(lag)
    future_df.dropna(inplace=True)


    future_df = future_df.drop(target_indicator, axis=1)
    future_df = future_df[X_train.columns]
    future_predictions = model.predict(future_df)

    print('INDEX: ', future_df.index)
    
    return model, predictions, mae, X_test.index, y_test, future_predictions, future_df.index

# Function to plot the results
def plot_results(country_name, target_indicator, df_prophet, forecast, predict_year, X_test_index, y_test, predictions, mae, future_predictions, future_index):
    fig, ax = plt.subplots(figsize=(12, 8))

    df_prophet['year'] = df_prophet['ds'].dt.year
    forecast['year'] = forecast['ds'].dt.year
    print(tabulate(df_prophet.head(1), headers='keys', tablefmt='pqsl'))
    X_test_index_year = X_test_index.year
    future_index_year = future_index.year
    
    # Plot actual data
    ax.plot(df_prophet['year'], df_prophet['y'], label='Actual Data', color='blue')
    
    # Plot Prophet predictions
    ax.plot(forecast['year'], forecast['yhat'], label='Prophet Predicted Data', color='red')
    
    # Plot Linear Regression predictions for the test set
    ax.plot(X_test_index_year, y_test, label='Test Data', color='green')
    ax.plot(X_test_index_year, predictions, label='Linear Regression Predictions', color='orange')
    
    # # Plot future predictions from Linear Regression
    # ax.plot(future_index, future_predictions, label='Future Linear Regression Predictions', color='purple', linestyle='--')
    
    # ax.axvline(x=pd.Timestamp(str(predict_year)), color='green', linestyle='--', label='Prediction Start Year')
    
    ax.set_title(f'{target_indicator} for {country_name} - MAE: {mae}')
    ax.set_xlabel('Year')
    ax.set_ylabel(target_indicator)
    ax.legend()
    plt.xticks(range(df_prophet['ds'].dt.year.min(), predict_year + 1, 5))  # Set x-axis ticks every 5 years
    plt.show()

# Main function to perform the analysis and prediction
def analyze_and_predict(file_path, country_name, target_indicator, predict_year):
    filtered_df = load_and_preprocess_data(file_path)
    country_pivot, df_prophet = prepare_data_for_country_indicator(filtered_df, country_name, target_indicator)
    forecast = generate_future_prophet_predictions(df_prophet, predict_year)
    model, predictions, mae, X_test_index, y_test, future_predictions, future_index = train_time_series_with_folds_autoreg_prophet_features(country_pivot, df_prophet, target_indicator, predict_year)
    plot_results(country_name, target_indicator, df_prophet, forecast, predict_year, X_test_index, y_test, predictions, mae, future_predictions, future_index)

# Example usage
file_path = 'WDI_CSV_2024_06_28/WDICSV.csv'
country_name = 'United States'
target_indicator = 'CO2 emissions from liquid fuel consumption (% of total)'
predict_year = 2030

analyze_and_predict(file_path, country_name, target_indicator, predict_year)
