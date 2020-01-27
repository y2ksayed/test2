#1/22 project due
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

movies = pd.read_csv(r'C:\Users\linds\.spyder-py3\imdb_1000.csv')
movies.head()
"""Check the number of rows and columns."""
movies.shape
#979 rows and 6 columns

""""Check the data type of each column."""
movies.dtypes
movies.columns.values.tolist()

"""Calculate the average movie duration."""
np.mean(movies.duration)
movies.duration.mean()

"""Sort the DataFrame by duration to find the shortest and longest movies"""
duration = movies.sort_values('duration')
duration.columns.values.tolist()
duration['title'].tail(1)
duration['title'].head(1)

"""Create a histogram of duration, choosing an "appropriate" number of bins."""
duration.duration.plot(kind='hist', bins=10);

"""Use a box plot to display that same data."""
duration.duration.plot(kind='box');

"""Intermediate level"""
"""Count how many movies have each of the content ratings."""
duration.groupby(['content_rating'])['title'].count()
content_rating = duration.groupby(['content_rating'])['title'].nunique()
content_rating = pd.DataFrame(content_rating)
content_rating['content_rating'] = content_rating.index
content_rating.reset_index(drop=True, inplace=True)
content_rating.columns = [ 'movie_count','content_rating']
"""Use a visualization to display that same data, including a title and x and y labels."""
x=content_rating.content_rating
x_pose = np.arange(len(x))
y=content_rating.movie_count
plt.figure(figsize=(15,8))
plt.bar(x_pose,y)
plt.ylabel('Movie Count')
plt.xticks(x_pose,x)
plt.title('Movie Count by Content Rating')
plt.show()

"""Convert the following content ratings to "UNRATED": NOT RATED, APPROVED, PASSED, GP."""
movies= movies.replace(["NOT RATED", "APPROVED", "PASSED", "GP"],
               ["UNRATED","UNRATED","UNRATED","UNRATED"])
"""Convert the following content ratings to "NC-17": X, TV-MA."""
movies= movies.replace(["X", "TV-MA"],
               ["NC-17","NC-17"])
"""Count the number of missing values in each column."""
movies.isna().sum()
movies.isnull().sum()

"""If there are missing values: examine them, then fill them in with "reasonable" values"""
movies["content_rating"] = movies.content_rating.fillna("reasonable")

"""Calculate the average star rating for movies 2 hours or longer, 
and compare that with the average star rating for movies shorter than 2 hours."""
avg_star_high = movies[movies["duration"] >= 120]
avg_star_low = movies[movies["duration"] < 120]

avg_star_high["star_rating"].mean()
avg_star_low["star_rating"].mean()

"""Use a visualization to detect whether there is a relationship between duration and star rating"""
dur_star = movies[["star_rating","duration"]].copy()
dur_star.corr()
#1
import seaborn as sns
sns.pairplot(dur_star)
#2
import matplotlib.pyplot as plt
plt.matshow(dur_star.corr())
plt.show()
#heatmap
dur_star_correlations = dur_star.corr();
sns.heatmap(dur_star_correlations);

"""Calculate the average duration for each genre"""
movies.dtypes
genre_avg = movies.groupby(['genre'])['duration'].mean()

"""Advanced level"""
"""Visualize the relationship between content rating and duration."""
con_rate_dur = movies[["content_rating","duration"]].copy()
con_rate_dur['content_rating'].unique()
#con_rate_dur = pd.get_dummies(con_rate_dur['content_rating'])
con_rate_dur = pd.concat([con_rate_dur, 
                          pd.get_dummies(con_rate_dur['content_rating'])],axis=1)
con_rate_dur.drop('content_rating', axis=1, inplace=True)

con_rate_dur.corr()
#1
import seaborn as sns
sns.pairplot(con_rate_dur)
#2
import matplotlib.pyplot as plt
plt.matshow(con_rate_dur.corr())
plt.show()
#heatmap
con_rate_dur_correlations = con_rate_dur.corr();
sns.heatmap(con_rate_dur_correlations);

"""Determine the top rated movie (by star rating) for each genre"""
movies.dtypes
top_rated=movies[["title","genre","star_rating"]].copy()
top_rated = movies.sort_values('star_rating')
top_rated.columns.values.tolist()
top_rated['genre'].head()
top_rated
#get the first row each group
top_rated.groupby('genre').first()

top_rated = top_rated.groupby(['genre'])['star_rating'].max()
content_rating = duration.groupby(['content_rating'])['title'].nunique()
content_rating = pd.DataFrame(content_rating)
content_rating['content_rating'] = content_rating.index
content_rating.reset_index(drop=True, inplace=True)
"""Check if there are multiple movies with the same title, and if so, 
determine if they are actually duplicates."""
movies[movies.duplicated()]
any(movies['title'].duplicated())
duplicates = movies[movies.duplicated('title', keep=False) == True]

"""Calculate the average star rating for each genre,
but only include genres with at least 10 movies"""
top_rated = top_rated.sort_values("star_rating",ascending = False)
top_ten = top_rated.groupby('genre').head(10).reset_index(drop=True)
top_ten = top_ten.sort_values('genre')
top_ten = top_rated.sort_values("star_rating",ascending = False)
top_ten = top_rated.sort_values(["genre","star_rating"],ascending = [True,False])

#more than 10 movies per genre
genre_counts = movies.genre.value_counts()
top_genres = genre_counts[genre_counts >= 10].index
movies[movies.genre.isin(top_genres)].groupby('genre').star_rating.mean()

# option 3: colculate the average star rating for all genres, then filter using a boolean Series
movies.groupby('genre').star_rating.mean()[movies.genre.value_counts() >= 10]

# option 4: aggregate by count and mean, then filter using the count
genre_ratings = movies.groupby('genre').star_rating.agg(['count', 'mean'])
genre_ratings[genre_ratings['count'] >= 10]

