### An exploration of a house price regressor using the Ames, Iowa dataset.

The inspiration for this project comes from attending a python meetup where a presenter did a quick overview of data science using the Ames, Iowa data set. [Ames, IA Housing Data (xls)](http://www.amstat.org/publications/jse/v19n3/decock/AmesHousing.xls)

As a recent graduate of the Galvanize immersive data science program I was exposed to many modeling techniques for which there wasn't enough time to delve into them deeper during the program.  So I took this as an opportunity to work on scraping more data and merging it in. I also did a few rounds of experimentation with feature transformations and ensemble modeling. 

##### The overall goal for this project was to make the highest performing model possible. There were many secondary goals for this project:
1. Add features over time in order to see how the error responded
1. Evaluate different models to see which ones perform the best
1. For highest performance try high performance modeling in the form of stacked ensemble.
1. Using EDA determine if certain features need feature engineering to perform the best
1. Incorporate data external to the provided data set as a useful real world exercise

##### 1: Tracking the rate of error over time

First, the metric used to evaluate error was RMSLE, Root Mean Squared Log Error.  RMSE is typical for models where the errors tend to be reasonably distributed. The _Log_ part is because of the log transformation that was performed on the price initially. This is done because the relative error is more useful than an absolute error.  

For example houses in this sample vary from 30k to over 500k. If the RMSE (for convenience the 'average error') is 25k that is pretty good for a 500k house (~5%) but it's a horrible error for a 30k house (80%+ error). Instead it's more useful to know something more akin to being 10% off on average.

In any case, using RMSLE is ideal in this case. In fact the Kaggle competition for this data set as well as the Zillow Challenge Home Price Regression also have their error in RMSLE.

From the following chart one can see how the rate of error changed as features were added to the data pipeline.

![Rate of Error Per Model Evolution](/imgs/error_over_time.png)

In general, until about Model Evolution 24 or so, we see the error decreasing as the models are provided more information. For a couple of evolutions the error remains flat then after evolution 26 the error starts creeping upwards again. I think that the models have started to overfit because there are more degrees of freedom and so the learning rates of the models probably need to be re-evaluated as the current state has implemented all current features.

##### A note on attempting to avoid overfitting
At the start of this project it was already known that the iterations, and therefore training runs, on this data would be very high. Therefore before typical train/test split an evaluation set was randomly selected using a fixed seed. This ensures that each time that the data is split the evaluation set remains the same. Before the project was started a limit of 8 evaluation scores was chosen in order to prevent overtraining to the evaluation set. The amount of error on the evaluation is a fair amount lower than the main model, probably because the larger set of data that is used for train/test has more extreme data points overall.

Therefore of the remaining 80% of data the data is split into 70% training set and 30% test set. Because the amount of error varied quite a bit 31 train/test splits are run and scored. For each evolution the median score is reported as well as the minimum and maximum score. This was done to approach a statistically signifigant range of error.

##### 2: Evaluate different models to see which ones perform the best

Currently SVM with RBF kernel typically performs the best and will very fast fitting/prediction. Gradient Boosting also usually does quite well but takes much longer to train/fit which is unsurprising since parallization doesn't help this model. Finally linear regression often performs quite horribly. Most likely this is due to an error in how the categorical features are converted to dummy columns. 

##### 3: For highest performance try high performance modeling in the form of stacked ensemble.

While this has been prototyped successfully in a Jupyter Notebook it is not currently implemented in the model.py class AmesModel. In the prototype a Random Forest, Linear Regression, and SVM-RBF were used for first stage predictions on 60% of the training data.  

For the 2nd stage of the model a Gradient Boosting Regressor was used upon the 3 predicted prices from the lower 3 models. Using the correct Y label the Gradient Booster was then trained to prefer the best model at different price ranges. This typically ended up reducing error by a decent amount, on the order of 10%. 

When this is implemented in the AmesModel class more specifics will be provided.

##### 4: Using EDA determine if certain features need feature engineering to perform the best

This stage of the project hasn't fully started yet. Some simplistic feature engineering has already been implemented. For example the predicted price of houses over 4000 ft^2 is to low by quite a bit. So a flag feature of "house_over_4000ft" was added which helped. Other transformations will probably be required to get optimal results and that will be a large part of the project moving forward now that all features have been implemented.

##### 5: Incorporate data external to the provided data set as a useful real world exercise

For extra data I simply went to the county website for Ames (Story county) and using the Parcel ID I gathered more info about the properties in the data set. Specifically which school district the home was in. So far this has mainly just been an exercise to scrape more data and merge it into the bigger set. As a feature this would probably be more useful if data about the schools assigned to the home (ie school rating) were then gathered and integrated. 

##### Future Plans

Current RMSLE is ~0.1203 so much work remains to be done.

After I started this project I learned that there is a Kaggle challenge with this data set. Therefore I plan to submit my model to the Kaggle challenge in order to evaluate it on the Kaggle leaderboards. 

The stacked ensemble has not yet been implemented in the primary model so that will be interesting to complete. Also instead of a gradient boosting predictor I plan to also evaluate a deep neural network as those tend to do very well with stacked models.
