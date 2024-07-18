import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Countries at risk of food insecurity


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
    'Fertilizer consumption (kilograms per hectare of arable land)',
    'Food production index (2014-2016 = 100)',
    'Crop production index (2014-2016 = 100)',
    'CO2 emissions from liquid fuel consumption (kt)',
    'CO2 emissions from gaseous fuel consumption (kt)',
    'CO2 emissions from solid fuel consumption (kt)',
    'Net migration',
    'Rural population (% of total population)',
    'Urban population (% of total population)',
    'Renewable internal freshwater resources, total (billion cubic meters)',
    'Average precipitation in depth (mm per year)'
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
  aggfunc='mean'
)

pivot_df_cleaned = pivot_df.dropna()

scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(pivot_df_cleaned), columns=pivot_df_cleaned.columns, index=pivot_df_cleaned.index)

kmeans = KMeans(n_clusters=10, random_state=42)
pivot_df_cleaned['Cluster'] = kmeans.fit_predict(scaled_df)

cluster_means = pivot_df_cleaned.groupby('Cluster')[['Cereal yield (kg per hectare)', 'Food production index (2014-2016 = 100)']].mean()
cluster_ranks = cluster_means.mean(axis=1).sort_values().index

colors = sns.color_palette("RdYlGn", n_colors=len(cluster_ranks))

cluster_color_map = {cluster: colors[i] for i, cluster in enumerate(cluster_ranks)}

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=pivot_df_cleaned, 
    x='Cereal yield (kg per hectare)', 
    y='Food production index (2014-2016 = 100)', 
    hue='Cluster', 
    palette=cluster_color_map
)
plt.title('Clusters of Countries Based on Agricultural and Environmental Indicators')
plt.xlabel('Cereal Yield (kg per hectare)')
plt.ylabel('Food Production Index (2014-2016 = 100)')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper right')
plt.show()

for cluster in cluster_ranks:
    cluster_rank = cluster_ranks.get_loc(cluster) + 1
    risk_level = 'High Risk' if cluster_rank <= len(cluster_ranks) // 2 else 'Low Risk'
    print(f'Rank {cluster_rank}, {risk_level}:')
    countries = pivot_df_cleaned[pivot_df_cleaned['Cluster'] == cluster].index.get_level_values('Country Name').unique()[:10]
    print(', '.join(countries))
    