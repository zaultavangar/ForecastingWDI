import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate



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

relevant_indicator_names = [
    'Cereal production (metric tons)',
    'Cereal yield (kg per hectare)',
    'CO2 emissions from liquid fuel consumption (kt)',
    'CO2 emissions from gaseous fuel consumption (kt)',
    'CO2 emissions from solid fuel consumption (kt)',
    'Food production index (2014-2016 = 100)',
    'Crop production index (2014-2016 = 100)',
    'Renewable internal freshwater resources, total (billion cubic meters)'
]

filtered_df = df[df['Indicator Name'].isin(relevant_indicator_names)]

melted_df = filtered_df.melt(
  id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
  var_name='Year',
  value_name='Value'
)

melted_df['Value'] = pd.to_numeric(melted_df['Value'], errors='coerce').round(3)

pivot_df = melted_df.pivot_table(
  values='Value',
  index=['Country Name', 'Year'],
  columns='Indicator Name',
  aggfunc='first'
)

pivot_df_cleaned = pivot_df.dropna()

print(tabulate(pivot_df_cleaned.head(20), headers='keys', tablefmt='psql'))

# Compute correlation matrix:
corr = pivot_df_cleaned.corr()

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.title('Correlation between CO2 Emissions and Agricultural Productivity')
plt.tight_layout()
plt.show()

print(melted_df.head(200))


