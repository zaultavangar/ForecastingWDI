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

relevant_indicator_names = [
    'Agricultural land (sq. km)',
    'Arable land (hectares)',
    'Arable land (hectares per person)',
    'Arable land (% of land area)',
    'Permanent cropland (% of land area)',
    'Land under cereal production (hectares)',
    'Cereal production (metric tons)',
    'Cereal yield (kg per hectare)',
    'CO2 emissions from liquid fuel consumption (kt)',
    'CO2 emissions from gaseous fuel consumption (kt)',
    'CO2 emissions from solid fuel consumption (kt)',
    'CO2 emissions from liquid fuel consumption (% of total)',
    'CO2 emissions from gaseous fuel consumption (% of total)',
    'CO2 emissions from solid fuel consumption (% of total)',
    'Renewable internal freshwater resources, total (billion cubic meters)',
    'Average precipitation in depth (mm per year)',
    'Renewable internal freshwater resources per capita (cubic meters)',
    'Adjusted savings: mineral depletion (current US$)',
    'Adjusted savings: mineral depletion (% of GNI)',
    'Fertilizer consumption (kilograms per hectare of arable land)',
    'Food production index (2014-2016 = 100)',
    'Crop production index (2014-2016 = 100)'
]

filtered_df = df[df['Indicator Name'].isin(relevant_indicator_names)]

melted_df = filtered_df.melt(
  id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
  var_name='Year',
  value_name='Value'
)

melted_df['Value'] = pd.to_numeric(melted_df['Value'], errors='coerce').round(3)

print(melted_df.head(200))

