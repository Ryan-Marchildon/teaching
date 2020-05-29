// ****************************************************
// Demonstration of a JS-Based Movie Recommender System 
// ****************************************************

// This script is used to prepare the data in .json format
// including feature preparation and data imputation. 

// Borrows code from this repo:
// https://github.com/javascript-machine-learning/movielens-recommender-system-javascript

import fs from 'fs';
import csv from 'fast-csv';

import prepareRatings from './preparation/ratings';
import prepareMovies from './preparation/movies';
import predictWithLinearRegression from './strategies/linearRegression';
import predictWithContentBased from './strategies/contentBased';
import { predictWithCfUserBased, predictWithCfItemBased } from './strategies/collaborativeFiltering';
import { getMovieIndexByTitle } from './strategies/common';


// *********************
// Extract Data from CSV 
// *********************
let MOVIES_META_DATA = {};
let MOVIES_KEYWORDS = {};
let RATINGS = [];

let ME_USER_ID = 0;

let moviesMetaDataPromise = new Promise((resolve) =>
  fs
    .createReadStream('./src/data/movies_metadata.csv')
    .pipe(csv({ headers: true }))
    .on('data', fromMetaDataFile)
    .on('end', () => resolve(MOVIES_META_DATA)));

let moviesKeywordsPromise = new Promise((resolve) =>
  fs
    .createReadStream('./src/data/keywords.csv')
    .pipe(csv({ headers: true }))
    .on('data', fromKeywordsFile)
    .on('end', () => resolve(MOVIES_KEYWORDS)));

let ratingsPromise = new Promise((resolve) =>
  fs
    .createReadStream('./src/data/ratings_small.csv')
    .pipe(csv({ headers: true }))
    .on('data', fromRatingsFile)
    .on('end', () => resolve(RATINGS)));

function fromMetaDataFile(row) {
  MOVIES_META_DATA[row.id] = {
    id: row.id,
    adult: row.adult,
    budget: row.budget,
    genres: softEval(row.genres, []),
    homepage: row.homepage,
    language: row.original_language,
    title: row.original_title,
    overview: row.overview,
    popularity: row.popularity,
    studio: softEval(row.production_companies, []),
    release: row.release_date,
    revenue: row.revenue,
    runtime: row.runtime,
    voteAverage: row.vote_average,
    voteCount: row.vote_count,
  };
}

function fromKeywordsFile(row) {
  MOVIES_KEYWORDS[row.id] = {
    keywords: softEval(row.keywords, []),
  };
}

function fromRatingsFile(row) {
  RATINGS.push(row);
}

console.log('Unloading data from files ... \n');

Promise.all([
  moviesMetaDataPromise,
  moviesKeywordsPromise,
  ratingsPromise,
]).then(init);

function init([ moviesMetaData, moviesKeywords, ratings ]) {
  // ****************
  // Data Preparation
  // ****************
  // -- Joins arrays
  // -- Maps categorical features to numerical values (e.g. one-hot encoding)
  // -- Imputes missing values based on feature means
  // -- Normalizes feature values

  const {
    MOVIES_BY_ID,
    MOVIES_IN_LIST,
    X,
  } = prepareMovies(moviesMetaData, moviesKeywords);




  // **************************
  // Save Prepared Data to Disk
  // **************************
  fs.writeFile("./src/data/MOVIES_BY_ID.json", JSON.stringify(MOVIES_BY_ID), 'utf8', function(err){
    if (err){
      console.log("An error occured while writing JSON object to file.");
      return console.log(err);
    }
    console.log('JSON file MOVIES_BY_ID has been saved.');
  });

  fs.writeFile("./src/data/MOVIES_IN_LIST.json", JSON.stringify(MOVIES_IN_LIST), 'utf8', function(err){
    if (err){
      console.log("An error occured while writing JSON object to file.");
      return console.log(err);
    }
    console.log('JSON file MOVIES_IN_LIST has been saved.');
  });

  fs.writeFile("./src/data/X.json", JSON.stringify(X), 'utf8', function(err){
    if (err){
      console.log("An error occured while writing JSON object to file.");
      return console.log(err);
    }
    console.log('JSON file X has been saved.');
  });

  fs.writeFile("./src/data/ratings.json", JSON.stringify(ratings), 'utf8', function(err){
    if (err){
      console.log("An error occured while writing JSON object to file.");
      return console.log(err);
    }
    console.log('JSON file ratings has been saved.');
  });

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