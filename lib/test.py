import pandas as pd
import numpy as np

data = pd.DataFrame({
    'Location.1': ['(34.0617, -118.2469)', '(25, -130)', '(0.0, 0.0)', '', '(34.0617, -118.2469)'],
    'AREA': ['1', '1', '1', '4', '5']

})
data[['Latitude', 'Longitude']] = data['Location.1'].str.extract(
    r'\(([^,]+), ([^)]+)\)').astype(float)

invalid_coords = (data['Latitude'] == 0.0) & (data['Longitude'] == 0.0)
print(data)
data.loc[invalid_coords, ['Latitude', 'Longitude']] = [np.nan, np.nan]

area_coords_mean: pd.DataFrame = data.groupby(
    'AREA')[['Latitude', 'Longitude']].transform('mean')

# Fülle fehlende Koordinaten basierend auf dem Durchschnitt der AREA
data[['Latitude', 'Longitude']] = data[[
    'Latitude', 'Longitude']].fillna(area_coords_mean)

print(data)
