# Pure-JS Movie recommender

Note: this code is a modified version of the recommendation system appearing here:
https://github.com/javascript-machine-learning/movielens-recommender-system-javascript


## Usage

* `npm install`

* `npm run-script data` to generate the data

* `npm start` to run the predictions script


### Notes on Input Data

We first populate the variables `MOVIES_META_DATA`, `MOVIES_KEYWORDS`, and `RATINGS` from a database (here we just have it stored in a CSV). 

These are the data files and contents:

`keywords.csv` (movie ID, plus all the keywords associated with it (each keyword is given its own id, eg. 392=england)

`movies_metadata.csv` (movie ID plus the following metadata: adult, belongs_to_collection, budget, genre, homepage, original_language, original_title, overview, popularity, poster_path, production_companies, production_countries, release_date, revenue, runtime, spoken_languages, status, tagline, title, video, vote_average, vote_count)

`ratings.csv`, userId, movieId, rating, timestamp
 