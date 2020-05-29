// ****************************************************
// Demonstration of a JS-Based Movie Recommender System 
// ****************************************************

// Borrows code from this repo:
// https://github.com/javascript-machine-learning/movielens-recommender-system-javascript

import fs from 'fs';

import prepareRatings from './preparation/ratings';
import prepareMovies from './preparation/movies';
import predictWithContentBased from './strategies/contentBased';
import { predictWithCfUserBased, predictWithCfItemBased } from './strategies/collaborativeFiltering';
import { getMovieIndexByTitle } from './strategies/common';


// *** NOTE ***
// This script demonstrates collaborative filtering and 
// content-based recommendation using cosine similarities, 
// implemented entirely within javascript. 

// In a production setting, we'd recommend that you implement your
// data pre-processing and recommendation steps with 
// a python backend, such as a simple Flask-based API, which 
// will be easier to code and is more performant. 


// *****************
// LOAD DATA OBJECTS
// *****************
console.log('Unloading data from files ... ');

let MOVIES_BY_ID = JSON.parse(fs.readFileSync('./src/data/MOVIES_BY_ID.json'));
let MOVIES_IN_LIST = JSON.parse(fs.readFileSync('./src/data/MOVIES_IN_LIST.json'));
let X = JSON.parse(fs.readFileSync('./src/data/X.json'));
let ratings = JSON.parse(fs.readFileSync('./src/data/ratings.json'));

console.log('Data loaded. \n');


// Add new user ratings
let ME_USER_ID = 0;
let ME_USER_RATINGS = [
  addUserRating(ME_USER_ID, 'Terminator 3: Rise of the Machines', '5.0', MOVIES_IN_LIST),
  addUserRating(ME_USER_ID, 'Jarhead', '4.0', MOVIES_IN_LIST),
  addUserRating(ME_USER_ID, 'Back to the Future Part II', '3.0', MOVIES_IN_LIST),
  addUserRating(ME_USER_ID, 'Jurassic Park', '4.0', MOVIES_IN_LIST),
  addUserRating(ME_USER_ID, 'Reservoir Dogs', '3.0', MOVIES_IN_LIST),
  addUserRating(ME_USER_ID, 'Men in Black II', '3.0', MOVIES_IN_LIST),
  addUserRating(ME_USER_ID, 'Bad Boys II', '5.0', MOVIES_IN_LIST),
  addUserRating(ME_USER_ID, 'Sissi', '1.0', MOVIES_IN_LIST),
  addUserRating(ME_USER_ID, 'Titanic', '1.0', MOVIES_IN_LIST),
];


// group the ratings by user and by movie
const {
  ratingsGroupedByUser,
  ratingsGroupedByMovie,
} = prepareRatings([ ...ME_USER_RATINGS, ...ratings ]);



/* ------------------------- */
//  Content-Based Prediction //
/* ------------------------- */
// We recommend titles based on similar movie metadata.  
// e.g. studio, director, actors, etc.

console.log('\n *********************');
console.log('(A) Content-Based Prediction');
console.log('(Ranks based on cosine similarity) \n');

contentPrediction('Batman Begins');
contentPrediction('Jurassic Park');
contentPrediction('Titanic');
contentPrediction('Back to the Future Part II');

function contentPrediction(title) {

  const contentBasedRecommendation = predictWithContentBased(
    X, MOVIES_IN_LIST, title
  );

  console.log(`\n Prediction based on "${title}": \n`);
  console.log(sliceTopRecommendations(
    contentBasedRecommendation, MOVIES_BY_ID, 10, true
  ));

}



/* ----------------------------------- */
//  Collaborative-Filtering Prediction //
//             User-Based              //
/* ----------------------------------- */

console.log('\n');
console.log('(C) Collaborative-Filtering (User-Based) Prediction ... \n');

console.log('(1) Computing User-Based Cosine Similarity \n');

const cfUserBasedRecommendation = predictWithCfUserBased(
  ratingsGroupedByUser,
  ratingsGroupedByMovie,
  ME_USER_ID
);

console.log('(2) Prediction \n');
console.log(sliceTopRecommendations(cfUserBasedRecommendation, MOVIES_BY_ID, 10, true));



/* ----------------------------------- */
//  Collaborative-Filtering Prediction //
//             Item-Based              //
/* ----------------------------------- */

console.log('\n');
console.log('(C) Collaborative-Filtering (Item-Based) Prediction ... \n');

console.log('(1) Computing Item-Based Cosine Similarity \n');

const cfItemBasedRecommendation = predictWithCfItemBased(
  ratingsGroupedByUser,
  ratingsGroupedByMovie,
  ME_USER_ID
);

console.log('(2) Prediction \n');
console.log(sliceTopRecommendations(cfItemBasedRecommendation, MOVIES_BY_ID, 10, true));

console.log('\n');
console.log('End ...');



// ------------------------------------
// ------------------------------------

// *****************
// HELPER FUNCTIONS
// *****************
export function addUserRating(userId, searchTitle, rating, MOVIES_IN_LIST) {
  const { id, title } = getMovieIndexByTitle(MOVIES_IN_LIST, searchTitle);

  return {
    userId,
    rating,
    movieId: id,
    title,
  };
}

export function sliceTopRecommendations(recommendations, MOVIES_BY_ID, count, onlyTitle) {
  recommendations = recommendations.filter(recommendation => MOVIES_BY_ID[recommendation.movieId]);

  recommendations = onlyTitle
    ? recommendations.map(mr => ({ title: MOVIES_BY_ID[mr.movieId].title, score: mr.score }))
    : recommendations.map(mr => ({ movie: MOVIES_BY_ID[mr.movieId], score: mr.score }));

  return recommendations
    .slice(0, count);
}

export function softEval(string, escape) {
  if (!string) {
    return escape;
  }

  try {
    return eval(string);
  } catch (e) {
    return escape;
  }
}