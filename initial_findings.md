## data_science_project
### Team Name: 100K-Offer
#### Member Names:
- Jie Lan 
- Hongjie Huang
- Bida Chen
- Runmin Lin
### Initial finding for submission 1
**Part 1**

In order to build the most suitable model for our data, we used cross validation to evaluate and then compare the performances of different models in terms of mean squared error. For the choice of our models, we picked Multiple Regression, Decision Tree Regression, Gradient Boosting Regression, and Random Forest Regression. Each of our team member is responsible for at least one test model, and while trying to obtain the optimal result, we tried to apply some basic deep learning into our model training procedure, thus we also build a Neural Network Regression. The following shows our resulted performances in term of mean squared error for each regression model:

Model | Mean Squared Error
------------ | -------------
Random Forest Regression | 997,822.6054296067
Gradient Boosting Regression | 885,245.202125923
Multiple Regression | 2,185,375.4933580006
Decision Tree Regression | 1,491,831.2221322171
Neural Network Regression | 827,503.6100231645


As the result implies, Gradient Boosting Regression provided with the best mean squared error which is 885,245.202125923. Worthy of remark, Neural Network Regression also had some outstanding performances when we were doing the test runs, but the overall result of it was really unstable, it varies around 800,000 to 1,300,000 in mean squared error, therefore we still decided to use Gradient Boosting Regression for the final prediction of test2. 
For all the features, according to the result of OLS Regression table and feature_importances from sci-kit learn, we can see that features such as size_sqft, bathrooms, bedroom, and zip_average_income (which is obtained from our external dataset) are driving the modeling performance, and for binary features, we’ve checked the coorelation for them, as the correlation table indicated, all binary features could highly affected the rents, thus we included all binary features when building our models.
<hr />

**Part 2**

To further improve the performances during modeling, we first appended test1 set to train set when we loaded the datasets, since this allow us to have a larger set which we can split it and conduct cross validation when evaluating models. During the data exploring and cleaning stages, we replaced missing data on floor number with the mode value of all floornumbers, replaced missing min_to_subway with the mean, and year_built with their mean as well, since we don’t want to just simply drop all of them. We also removed outliers in our dataset and then encoded categorical features and drop useless features to make the modeling process clean. Then our strategies to improve the prediction is to tune the hyperparameters for tree based models in order to achieve a better result during the learning process, and we also plotted the learning curve. For linear model which is Multiple Regression in our case, we inspected the OLS Regression result and removed some useless features to improve the performances. Lastly, as we mentioned in part 1, we compared all the performances, and then apply Gradient Boosting Regression which has the best mean squared error to prediction against test2, and output our final predictions with their corresponding rental_id to test2_prediction.csv .
