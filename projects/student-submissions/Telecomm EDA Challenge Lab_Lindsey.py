#Day4 Homework
#Lindsey Bang
#Telecom
#Meta Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import geopandas as gpd
#from shapely.geometry import Point, Plygon
meta = r'C:\Users\linds\.spyder-py3\metadata.csv'
df = pd.read_csv(meta)
df.dtypes
df.describe()
df.head()
df.shape

#Comm Type
df.groupby("Comm Type")["Comm Type"].count()
df.groupby("Comm Type")["Comm Type"].count().reset_index(name='count').sort_values(['count'], ascending=False)
#Cell Tower Location
Cell_tower_location_count = df.groupby("Cell Tower Location")["Cell Tower Location"].count().reset_index(name='count').sort_values(['count'], ascending=False)
import os
os.environ['epsg'] = r'C:\Users\linds\Anaconda3\Library\share\basemap'
from mpl_toolkits.basemap import Basemap
import mpl_toolkits
from geopy import geocoders
from geopy.extra.rate_limiter import RateLimiter
#geocoding-in-python-get-latitude-and-longitude-from-addresses-using-api-key
from geopy import geocoders
from geopy.geocoders import GoogleV3

API_KEY = os.getenv("API1234")
g = GoogleV3(api_key=API_KEY)

from urllib2 import urlopen
import json

#
#https://stackoverflow.com/questions/44488167/plotting-lat-long-points-using-basemap
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import mpl_toolkits

#
#https://stackoverflow.com/questions/44488167/plotting-lat-long-points-using-basemap
from bokeh.plotting import figure, show
from bokeh.sampledata.us_states import data as states
from bokeh.models import ColumnDataSource, Range1d

meta = r'C:\Users\linds\.spyder-py3\metadata.csv'
df = pd.read_csv(meta)
lat = df['Latitude'].values
long = df['Longitude'].values



# determine range to print based on min, max lat and long of the data
margin = .2 # buffer to add to the range
lat_min = min(lat) - margin
lat_max = max(lat) + margin
long_min = min(long) - margin
long_max = max(long) + margin

# create map using BASEMAP
m = Basemap(llcrnrlon=long_min,
            llcrnrlat=lat_min,
            urcrnrlon=long_max,
            urcrnrlat=lat_max,
            lat_0=(lat_max - lat_min)/2,
            lon_0=(long_max-long_min)/2,
            projection='merc',
            resolution = 'h',
            area_thresh=10000.,
            )
m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color = 'white',lake_color='#46bcec')
# convert lat and long to map projection coordinates
lons, lats = m(long, lat)
# plot points as red dots
m.scatter(lons, lats, marker = 'o', color='r')
plt.show()

# create map using Bokeh
source = ColumnDataSource(data = dict(lat = lat,lon = long))
# get state boundaries
state_lats = [states[code]["lats"] for code in states]
state_longs = [states[code]["lons"] for code in states]

p = figure(
           toolbar_location="left",
           plot_width=1100,
           plot_height=700,
           )

# limit the view to the min and max of the building data
p.y_range = Range1d(lat_min, lat_max)
p.x_range = Range1d(long_min, long_max)
p.xaxis.visible = False
p.yaxis.visible = False
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

p.patches(state_longs, state_lats, fill_alpha=0.0,
      line_color="black", line_width=2, line_alpha=0.3)

p.circle(x="lon", y="lat", source = source, size=4.5,
         fill_color='red',
         line_color='grey',
         line_alpha=.25
         )
show(p)

#geo pandas
#https://towardsdatascience.com/geopandas-101-plot-any-data-with-a-latitude-and-longitude-on-a-map-98e01944b972

import pandas as pd
import matplotlib.pyplot as plt
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
%matplotlib inline

import numpy as np

#need to download the map
#The first step is to download a shape-file(.shp file) of that area 
#if you know the general area in which your geo-spatial data exists. 
#If you don’t know where to find a shape-file, Google it! 
#There are often a couple of other files that come with the shape-file; 
#make sure to keep all of those files in the same folder together, 
#or you won’t be able to read in your file.
street_map = gpd.read_file(r'C:\Users\linds\.spyder-py3\metadata.csv')
fig,ax = plt.subplots(figsize = (15,15))
streer_map.plot(ax=ax)

#data
meta = r'C:\Users\linds\.spyder-py3\metadata.csv'
df = pd.read_csv(meta)
crs = {'init': 'epsg:4326'}
df.head()

geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
geometry[:3]

geo_df = gpd.GeoDataFrame(df, crs = crs, geometry = geometry)
geo_df.head()

fig,ax = plt.subplots(figsize = (15,15))
street_map.plot(ax=ax, alpha=0.4, color = "grey")
geo_df.plot(ax=ax, markerszie = 20, color="blue", marker ="o", label="Neg")
plt.legend(prop={"size"=15})






