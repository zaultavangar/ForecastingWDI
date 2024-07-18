import pandas as pd
from tqdm import tqdm

file_path = 'WDI_CSV_2024_06_28/WDICSV.csv'

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

# filter out indicators that start with "Population"
filtered_indicators = sorted_indicators[~sorted_indicators['Indicator Name'].str.startswith(("Population", "Age dependency"))]

# drop rows with counts less than 5000
filtered_indicators = filtered_indicators[filtered_indicators['Count'] >= 5000]

filtered_indicators = filtered_indicators.reset_index(drop=True)
filtered_indicators.index += 1


print(filtered_indicators.head(250))
print("----------------------------------")
print("Total rows after filtering:", len(filtered_indicators))