# Housing Price Prediction Model

The goal of this project is to predict home prices based on 12 different features (area, bedrooms, bathrooms, etc.) utilizing sci-kitlearn's linear regression library. There will be another implementation done manually. 

## General

I chose to use a multilinear regression model because its purpose is to predict one dependent variable, **price**, based on multiple independent variables like mentioned above. 

To ensure that I had a reasonable output for this model on the training set, I had to adjust many parameters in the model to accurately derive a proper mean squared error (MSE). 

There were three components to the model I adjusted for accuracy with assistance from ChatGPT:
    
1. Logarithmic adjustments to price
2. One-hot encoding with only binary categorical values
3. Keep data normalized
4. Remove unnecessary features

The orientation of the dataset mimicked a logarithmic structure, and to ensure the MSE would have a reasonable value, I applied a logarithmic shift to every datapoint in the dataset for price. 

Additionally, some values in the dataset are not numerical, so a one-hot encoding splits some features into separate columns with binary values that are easily consumable by the model. Some features have multiple values like furnishing (furnished, semi-furnished, and unfurnished). By dropping the first value, if both semi-furnished and unfurnished are 0, it implies that the house is furnished.

It is imperative in this model to keep data normalization because without it the model would output an extremely high MSE. But by normalizing the data we can have a more linear relationship between the predicted values and the actual values.

Lastly, the mean absolute percentage error was significantly high, inferring something about the model was inaccurate. Although there is a smaller training set, all the features in the data set may not have as much correlation to the price as originally thought. Also added feature engineering.

The dataset used in this project was found on Kaggle and is free to use. It contains roughly 500 samples of housing data for training and testing. 
    
Link: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset?resource=download


Discuss:

    Why you chose the model that you did
    Adjustments made (log skewing, one-hot encoding with no multicolinearity) 
    MSE issues
    Red line ? 