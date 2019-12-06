## data_science_project
### Team Name: 100K-Offer
#### Member Names:
- Jie Lan 
- Hongjie Huang
- Bida Chen
- Runmin Lin

### Project Findings for Questions and Tasks section

## Data Usage.

Our team researched about various topics for factors that affect the rental price beside the ones that 
provided in the dataset. The first factor that came to our mind was the average income by zipcode, because 
it directly represents peoples' economic strength in that area, so it is reasonable to assume that the 
area that has a higher average income tend to have a higher rental price, and the correlation between average income based on the zipcode and rent is 0.393228, which is significant.

The other factor is the unemployment rate, people tend to stay in the areas that have more working opportunities, therefore we can assume that the areas with high unemployment rate are more likely to have lower rent. We found that the correlation between zip_ unemployment _rate and rent is -0.167267 which clearly shows that the unemployment rate and rent has a negative correlation, which indicates that area with higher unemployment rate tend to have a lower rent, but the relationship is not that significant comparing to the average income.

We also tried several other dataset including the Population dataset from NYC OpenData, housing sale price dataset from kaggle and median estimated rental values dataset from Zillow since population of the neighborhood, housing prices and median rental price of the zipcode could possibly affect our the estimated rental. However, not all of them are improving our performance, so we decided to leave only the xxx.

## Data Exploration.

As we have stated in the initial finding, there are some outliers that are not reasonable in real world scene such as rent,bathrooms,size_ sqft,year_ built,min_ to_subway and floor number, so we set up several boundaries to each element to eliminate these outliers. 

Missing values can be a challenge for our analysis, because if we just simply drop all of them, our MSE will be affect significantly. Therefore, we replaced missing data on floor number with the mode value of all floor numbers, replaced missing min_to_subway with the mean value, and year_built with their mean as well. 

Some features like the addr_unit, building_id and so on can be problematic since they are not directly relate to the rent that we are predicting, so we encoded categorical features and drop useless features to make the modeling process clean. 

We have plotted the scatter plots for the continuous_features in the data cleaning section.

[https://github.com/JiejayLan/data_science_project/blob/master/main.ipynb](https://github.com/JiejayLan/data_science_project/blob/master/main.ipynb)

## Transformation and Modeling.

As we have stated in the initial findings,according to the result of OLS Regression table and feature_importances from sci-kit learn, we can see that features such as size_sqft, bathrooms, bedroom, and zip_ average_ income (which is obtained from our external dataset) are driving the modeling performance because generally people cares about the size and bathrooms/bedroom of the house and zip average income represent average people's affordability, and for binary features, we’ve checked the correlation for them, as the correlation table indicated, all binary features could highly affected the rents, thus we included all binary features when building our models.

One important feature or modification that we had is to append test1 set to train set when we loaded the datasets, since this allow us to have a larger set which we can split it and conduct cross validation when evaluating models. The other important feature is the turning of the hyperparameters for tree based models. By using functions that help us find the better parameters, we did achieve better results.

For the implementation of our models, we followed the same format as provided in the lecture, where we used cross validation to evaluate each model and then compare the performances of different models in terms of mean squared error in order to build the most suitable model for our data. For the choice of our models, we picked Multiple Regression, Decision Tree Regression, Gradient Boosting Regression, and Random Forest Regression. 

For each model we set up parameters according to the nature of the model and apply tuning hyper parameters if possible, so that we will get reasonably good result from each model. In order to obtain the optimal result, we tried to apply some basic deep learning into our model training procedure, thus we also build a Neural Network Regression which gave outstanding result. However, due to the fact that Neural Network Regression is not stable, we chose the Gradient Boosting Regression instead. We think this type of model works well because it builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.Therefore, in our case, it is more stable and minimize the errors that we made comparing to other models that we used.