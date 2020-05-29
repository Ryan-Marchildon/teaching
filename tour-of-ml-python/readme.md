# A Tour of Machine Learning in Python

*The basics of how to perform exploratory data analysis and build a machine learning pipeline.*

In this tutorial notebook (`tour-of-basic-ml.ipynb`) I demonstrate key elements and design approaches that go into building a well-performing machine learning pipeline. The topics I’ll cover include:

1. Exploratory Data Analysis and Feature Engineering.
2. Data Pre-Processing including cleaning and feature standardization.
3. Dimensionality Reduction with Principal Component Analysis and Recursive Feature Elimination.
4. Classifier Optimization via hyperparameter tuning and Validation Curves.
5. Building a more powerful classifier through Ensemble Voting and Stacking.

Along the way we’ll be using several important Python libraries, including scikit-learn and pandas, as well as seaborne for data visualization.

Our task in this tutorial is a binary classification problem inspired by Kaggle’s “Getting Started” competition, Titanic: Machine Learning from Disaster. The goal is to accurately predict whether a passenger survived or perished during the Titanic’s sinking, based on data such as passenger age, class, and sex. 

I have taken my time here to focus on fundamentals that should be a part of every data scientist’s toolkit, namely on the data exploration and pre-processing. We also touch on how to train and tune 'classical' machine learning models through the scikit-learn library. 

**Note to github users** you can view a pre-rendered version of this notebook [here](http://rpmarchildon.com/ai-titanic/).