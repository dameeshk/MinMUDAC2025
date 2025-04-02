import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import os
from pathlib import Path
import numpy as np

# Get the current directory and construct paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'Data')
filled_data_path = os.path.join(parent_dir, 'county filler', 'filled_data.xlsx')

# Read the Excel file with match data
df = pd.read_excel(filled_data_path)

# Count unique matches by county using Match ID
county_counts = df.groupby('Big County')['Match ID 18Char'].nunique()
matches_data = pd.DataFrame(county_counts).reset_index()
matches_data.columns = ['NAME', 'total_matches']

# Load Minnesota county boundaries
shapefile_path = os.path.join(data_dir, 'tl_2020_us_county.shp')
mn_counties = gpd.read_file(shapefile_path)
mn_counties = mn_counties[mn_counties['STATEFP'] == '27']  # Filter for Minnesota counties

# Clean county names to match our data
def clean_county_name(name):
    if pd.isna(name):
        return None
    return name.replace(' County', '').strip()

# Clean county names in both datasets
mn_counties['NAME'] = mn_counties['NAME'].apply(clean_county_name)
matches_data['NAME'] = matches_data['NAME'].apply(clean_county_name)

# Merge the match data with the county geometries
mn_counties = mn_counties.merge(
    matches_data,
    on='NAME',
    how='left'
)

# Fill NaN values with 0
mn_counties['total_matches'] = mn_counties['total_matches'].fillna(0)

# Create the visualization with a more square aspect ratio
fig, ax = plt.subplots(1, 1, figsize=(10, 12))

# Set scale to 1800 to properly show Hennepin's 1768 matches in darkest color
max_scale = 1800

# Create custom boundaries that emphasize the data distribution
# Adjusted to ensure Hennepin (1768) falls in the highest category
bounds = [0, 25, 50, 100, 200, 400, 700, 1000, 1500, 1800]
colors = ['#FFFFFF', '#FFF5BA', '#FFE298', '#FFB861', '#FF8A3C', '#FF5C00', '#FF2B00', '#CC0000', '#990000', '#660000']

# Create custom colormap and normalization
custom_cmap = LinearSegmentedColormap.from_list('custom_YlOrRd', colors)
norm = BoundaryNorm(bounds, custom_cmap.N)

# Plot the counties with adjusted parameters
mn_counties.plot(
    column='total_matches',
    ax=ax,
    legend=True,
    legend_kwds={
        'label': 'Total Number of Matches\nby County',
        'orientation': 'vertical',
        'shrink': 0.7,
        'fraction': 0.046,
        'pad': 0.04,
        'location': 'right',
        'boundaries': bounds,
        'norm': norm,
        'format': '%1.0f'
    },
    missing_kwds={'color': 'lightgrey'},
    cmap=custom_cmap,
    edgecolor='black',
    linewidth=0.5,
    norm=norm
)

# Remove axes
ax.axis('off')

# Add title
plt.title('Big Brothers Big Sisters Matches\nby County', 
          pad=20, size=14)

# Add legend for top 5 counties in a box
top_5_counties = matches_data.nlargest(5, 'total_matches')
legend_text = 'Top 5 Counties:\n'
for _, row in top_5_counties.iterrows():
    legend_text += f"{row['NAME']}: {int(row['total_matches'])}\n"

# Position the text box in the lower right, outside the map
plt.figtext(0.75, 0.15, legend_text,
            fontsize=8,
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
            ha='left',
            va='bottom')

# Add magnifying glass icon
plt.figtext(0.92, 0.12, '🔍', fontsize=12)

# Adjust layout to prevent legend overlap
plt.tight_layout()

# Save the map
output_path = os.path.join(current_dir, 'minnesota_matches.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Map has been generated and saved as '{output_path}'")
print(f"Scale set to: 0-{max_scale}")

# Print top 5 counties for reference
print("\nTop 5 counties by total number of matches:")
for _, row in top_5_counties.iterrows():
    print(f"{row['NAME']}: {int(row['total_matches'])}") 