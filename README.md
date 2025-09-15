Medium Article Link - https://medium.com/@alekyagudise1008/data-science-walkthrough-crisp-dm-analysis-of-imdb-top-250-movies-3a4f7bbdc70b

ChatGPT Link - https://chatgpt.com/share/68c76c3a-4494-8011-ae70-51753013cf35

# IMDb Top 250 Movies — CRISP-DM Project

This project applies the CRISP-DM methodology to analyze the IMDb Top 250 Movies dataset, exploring data preparation, modeling, evaluation, and deployment.

## Dataset
- Source: Kaggle — IMDb Top 250 Movies
- Size: 250 movies, 12+ features
- Target variable: IMDb Rating

###  Key Features
- **Name**: Movie title  
- **Year**: Release year  
- **Runtime**: Duration of the film (hours & minutes, later converted to minutes)  
- **Certificate**: Rating category (e.g., PG, R, etc.)  
- **Genre**: One or more genres (Drama, Crime, Action, etc.)  
- **Directors**: Film director(s)  
- **Budget**: Estimated production budget  
- **Box Office**: Worldwide gross revenue  
- **Rating**: IMDb rating (target variable)

## 📊 Exploratory Data Analysis

### Runtime Distribution
![Runtime Distribution](outputs/runtime_distribution.png)

### Feature Correlation Heatmap
![Feature Correlation](outputs/correlation_heatmap.png)


## 📈 Model Evaluation

### Model Comparison (RMSE)
![RMSE Comparison](outputs/model_comparison_rmse.png)

### Model Comparison (R² Scores)
![R2 Comparison](outputs/model_comparison_r2.png)

###  Summary Statistics
- **Runtime**: Average ~129 minutes, range 40–240 minutes (after capping)  
- **Budget**: Ranges from under $1M to over $2B (log-transformed for skewness)  
- **Box Office**: Highly skewed, with blockbusters grossing >$1B  
- **Ratings**: Range from 8.0 to 9.3, clustered tightly since all films are highly acclaimed  
- **Genres**: Drama is the most frequent genre, followed by Crime and Action  
- **Directors**: Several directors (e.g., Christopher Nolan, Steven Spielberg, Martin Scorsese) appear multiple times  

###  Insights
- Longer runtime, higher budgets, and strong box office returns tend to align with higher ratings.  
- Director reputation (measured by the number of Top 250 films they directed) is a significant predictor.  
- Genre has a smaller influence, since most movies in the dataset already fall into critically strong categories.  

This summary forms the basis for the **Data Preparation** and **Modeling** phases in the CRISP-DM methodology.

