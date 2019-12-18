#Lindsey Bang
#homework
#rock

import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
#1-------------------------------------------------------------------------------------------------
#read CSV file
# Load the data.
rockfile = pd.read_csv('.spyder-py3/rock.csv')
#load the data
rockfile.head()
# Look at the information regarding its columns.
rockfile.columns
#2-------------------------------------------------------------------------------------------------
# Change the column names when loading the '.csv':
rock_cols=['Song Title','Artist','Album Release Year','Combination','First','Year','PlayCounts','f*g']
rockfile= pd.read_csv('.spyder-py3/rock.csv',header=0,names=rock_cols)
# Change the column names using the `.rename()` function.
rockfile = rockfile.rename(columns={'Song Title':'SongTitle','Artist':'RockArtist','Album Release Year':'ReleaseYear','Combination':'Combo','First':'1st','Year':'YYYY','PlayCounts':'PlayCount','f*g':'F*G'})
# Replace the column names by reassigning the `.columns` attribute.
rockfile.columns = rock_cols
#3-------------------------------------------------------------------------------------------------
# Show records where df['release'] is null
rockfile['ReleaseYear'].isnull()
rockfile['ReleaseYear'].isnull().sum()
#4-------------------------------------------------------------------------------------------------
# Replace release nulls with 0
rockfile.ReleaseYear.fillna(value=0)
rockfile.ReleaseYear.fillna(value=0, inplace=True)
releaseyear = rockfile.ReleaseYear
#Verify that release contains no null values
rockfile.ReleaseYear.isnull().sum(axis=0)
#5-------------------------------------------------------------------------------------------------
#A Look at the data types for the columns. Are any incorrect given what the data represents
rockfile.dtypes
#5-------------------------------------------------------------------------------------------------
#Figure out what value(s) are causing the release column to be encoded as a string instead of an integer
rockfile.loc[:,'ReleaseYear']
rockfile['ReleaseYear'] = rockfile['ReleaseYear'].astype(int)
#error row -> 'SONGFACTS.COM'
#replace
rockfile.ReleaseYear.replace('SONGFACTS.COM', 0, inplace=True)
#string to interger
rockfile['ReleaseYear'] = rockfile['ReleaseYear'].astype(int)
rockfile.ReleaseYear.dtypes
#7-------------------------------------------------------------------------------------------------
#the latest release date
print(rockfile.ReleaseYear.sort_values(ascending=False).head(1))
#the earliest release date
mask = (rockfile.ReleaseYear > 0)
print(rockfile[mask].ReleaseYear.sort_values(ascending=True).head(1))
print(rockfile[rockfile.ReleaseYear > 0].ReleaseYear.sort_values(ascending=True).head(1))
#Based on the summary statistics, is there anything else wrong with the release column
#summary
rockfile.info()
rockfile.columns
rockfile.groupby('RockArtist').ReleaseYear.agg(['count', 'mean', 'min', 'max']).sort_values('mean')
rockfile.describe()
rockfile.ReleaseYear.describe()
#8-------------------------------------------------------------------------------------------------
#print out the song, artist, and whether or not the release date is < 1970
rock_filter = rockfile[(rockfile.ReleaseYear <1970) &  (rockfile.ReleaseYear!= 0 )]
rock_filter
function = (rockfile.ReleaseYear <1970) &  (rockfile.ReleaseYear!= 0 )
rock_file_1 = rock_filter[['ReleaseYear','SongTitle','RockArtist']].sort_values('ReleaseYear')
#apply() function, apply the function you wrote to the first four rows of the DataFrame
rockfile['under1970'] = rockfile.apply(lambda row: row.ReleaseYear < 1970 and row.ReleaseYear != 0, axis=1)
rockfile.head()
#9-------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
df_cols=['SongTitle','Artist','ReleaseYear','Combination','First','Year','PlayCounts','F*G']
df = pd.read_csv('.spyder-py3/rock.csv',header=0,names=df_cols)
df.dtypes
df['ReleaseYear'] = df['ReleaseYear'].astype(int)
df.ReleaseYear.replace('SONGFACTS.COM', 0, inplace=True)
df.ReleaseYear.isnull().sum(axis=0)
df.ReleaseYear.fillna(value=0, inplace=True)
type(1)
#A Write the function that takes a column and converts all of its values to float if possible and np.nan
#otherwise. The return value should be the converted Series
try:
    convert_float = lambda x: x.astype(float)
except:
    np.nan
df.dtypes

#Try your function out on the rock song data and ensure the output is what you expected
df['ReleaseYear']=convert_float(df['ReleaseYear'])
df['PlayCounts']=convert_float(df['PlayCounts'])
df['First']=convert_float(df['First'])
df['Year']=convert_float(df['Year'])
df['F*G']=convert_float(df['F*G'])

#print(pd.DataFrame(map(convert_float, df['PlayCounts'])))
df.head(10)

#del df['output']
print(df.head())
#Describe the new float-only DataFrame.
df.describe()
df.dtypes
#9 with a different approach----------------------------------------------------------------
#9-------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
df_cols=['SongTitle','Artist','ReleaseYear','Combination','First','Year','PlayCounts','F*G']
df = pd.read_csv('.spyder-py3/rock.csv',header=0,names=df_cols)
df.dtypes
df['ReleaseYear'] = df['ReleaseYear'].astype(int)
df.ReleaseYear.replace('SONGFACTS.COM', 0, inplace=True)
df.ReleaseYear.isnull().sum(axis=0)
df.ReleaseYear.fillna(value=0, inplace=True)

#A Write the function that takes a column and converts all of its values to float if possible and np.nan
#otherwise. The return value should be the converted Series

def converter_helper(value):
    try:
        return float(value)
    except:
        return np.nan
def convert_to_float(column):
    column = column.map(converter_helper)
    return column

#Try your function out on the rock song data and ensure the output is what you expected
df.apply(convert_to_float).head(10)
df2 = df.apply(convert_to_float)
#print(pd.DataFrame(map(convert_float, df['PlayCounts'])))
#Describe the new float-only DataFrame.
df2.describe()
df2.dtypes