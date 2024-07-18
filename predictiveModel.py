import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

## IN PROGRESS

# Load and preprocess data
def load_and_preprocess_data(file_path):
    pd.set_option('display.max_rows', 250)
    pd.set_option('display.max_columns', None)
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

    # Select the target indicator and rename columns for Prophet
    df_prophet = country_pivot[[target_indicator]].reset_index()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'].astype(str) + '-01-01')  # Convert Year to datetime
    
    return df_prophet

# Function to train and predict using the model
def train_and_predict(df_prophet, predict_year):
    model = Prophet()
    model.fit(df_prophet)

    max_year = df_prophet['ds'].dt.year.max()

    # Create future dataframe
    future = model.make_future_dataframe(periods=predict_year - max_year + 1, freq='Y')
    
    # Predict future values
    forecast = model.predict(future)
    
    return forecast, future['ds']

# Function to plot the results
def plot_results(country_name, target_indicator, df_prophet, forecast, predict_year):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot actual data
    ax.plot(df_prophet['ds'], df_prophet['y'], label='Actual Data', color='blue')
    
    # Plot predictions
    ax.plot(forecast['ds'], forecast['yhat'], label='Predicted Data', color='red')
    
    # Highlight future predictions
    ax.axvline(x=pd.Timestamp(str(predict_year)), color='green', linestyle='--', label='Prediction Start Year')
    
    ax.set_title(f'{target_indicator} for {country_name}')
    ax.set_xlabel('Year')
    ax.set_ylabel(target_indicator)
    ax.legend()
    plt.xticks(range(df_prophet['ds'].dt.year.min(), predict_year + 1, 5))  # Set x-axis ticks every 5 years
    plt.show()

    

# Main function to perform the analysis and prediction
def analyze_and_predict(file_path, country_name, target_indicator, predict_year):
    filtered_df = load_and_preprocess_data(file_path)
    df_prophet = prepare_data_for_country_indicator(filtered_df, country_name, target_indicator)
    forecast, future_years = train_and_predict(df_prophet, predict_year)
    plot_results(country_name, target_indicator, df_prophet, forecast, predict_year)

# Example usage
file_path = 'WDI_CSV_2024_06_28/WDICSV.csv'
country_name = 'United States'
target_indicator = 'CO2 emissions from liquid fuel consumption (% of total)'
predict_year = 2030

analyze_and_predict(file_path, country_name, target_indicator, predict_year)
