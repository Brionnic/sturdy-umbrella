##### An exploration of a house price regressor using the Ames, Iowa dataset.

The inspiration for this project comes from attending a python meetup where a presenter did a quick overview of data science using the Ames, Iowa data set. [Ames, IA Housing Data (xls)](http://www.amstat.org/publications/jse/v19n3/decock/AmesHousing.xls)

As a recent graduate of the Galvanize immersive data science program I was exposed to many modeling techniques for which there wasn't enough time to delve into them deeper during the program.  So I took this as an opportunity to work on scraping more data and merging it in. I also did a few rounds of experimentation with feature transformations and ensemble modeling. 

##### The overall goal for this project was to make the highest performing model possible. There were many secondary goals for this project:
1. Add features over time in order to see how the error responded
1. Evaluate different models to see which ones perform the best
1. For highest performance try high performance modeling in the form of stacked ensemble.
1. Using EDA determine if certain features need feature engineering to perform the best
1. Incorporate data external to the provided data set as a useful real world exercise

##### 5: Incorporate data external to the provided data set as a useful real world exercise

For extra data I simply went to the county website for Ames (Story county) and using the Parcel ID I gathered more info about the properties in the data set. Specifically which school district the home was in. So far this has mainly just been an exercise to scrape more data and merge it into the bigger set. As a feature this would probably be more useful if data about the schools assigned to the home (ie school rating) were then gathered and integrated.

The current result being a stacked ensemble that consists of 4 home price regressors for the first stage and then 1 gradient boosting regressor that takes the predicted prices from the lower models as the input and uses that to make the final prediction. 

Current RMSLE is ~0.1203 so much work remains to be done.

This project is currently a work in progress and documention will improve quite a bit.
