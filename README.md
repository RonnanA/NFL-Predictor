# NFL Game Predictor
NFL Game Predictor. A data-driven machine learning project that predicts the outcome of an NFL matchup using team performance statistics from previous seasons. 

It scrapes, cleans, and analyzes historical game data, then trains a predictive model that can predict the probability of a home team winning any given matchup.

## Project Overview
This project aims to answer the question:
> *Given two NFL teams what are the chances the home team wins?*
It uses a combination of:
- Web Scraping (automated data collection from Pro Football Reference)
- Data processing and feature engineering
- Machine learning models such as RandomForestClassifier, and plans for XGBoost

to create a system that learns from historical patterns and predicts future games.

## How It Works
### Data Collection
The custom built scraper automatically gathers:
- Every regular-season NFL game context for any modern era season
- Detailed team stats (passing/rushing yards, turnovers, conversions, etc.)

The scraper saves the two datasets as Season data and Stats data. These are merged into a single master CSV file per season.

### Data Cleanup
Before training, the data such as postseason games, non-numeric columns, or irrelevant metadata, are removed or converted. The result is a clean, structured dataset ready for modeling.

### Feature Engineering
The project computes rolling averages for each team's key performance metrics, which represents each team's *recent form*.

Example features include:
- `diff_total_yards_avg`
- `diff_turnovers_avg`
- `diff_pass_yds_avg`
- `diff_rush_yds_avg`

### Model Training
The model is currently being trained using scikit-learn pipelines, and using the RandomForestClassifier algorithm. The project supports to training modes:
- Evaluation Mode: intended for measuring and improving model accuracy and has access to select evaluation metrics
- Predict Mode: trains on all available data to produce a deployable model for future predictions

### Making Predictions
Once trained, the model can predict the probability of a home team winning a **future matchup** (e.g., TB @ BUF) using the teams' most recent form data.

Example output:
`TB @ BUF -> predicted home win probability: 51.0%`

### Project Goals
- Learn and apply effective feature engineering techniques
- Demonstrate an end-to-end ML workflow on real sports data
- Produce a model that updates automatically as new games are played
<br><br><br>

## --README Will Be Updated As Project Progresses--