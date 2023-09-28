# import python packages 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the data
data = pd.read_csv('ames_housing_price_dataset.csv')

# Prepare the data
data = data.dropna()
data = data.select_dtypes(exclude=['object'])
data = pd.get_dummies(data)

# Create new features
data['avg_price_neighborhood'] = data['SalePrice'].groupby(data['Neighborhood']).transform('mean')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('SalePrice', axis=1), data['SalePrice'], test_size=0.25)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
r2_score = model.score(X_test, y_test)

# Print the results
print('R-squared score:', r2_score)

# Predict the price 
house_features = {
    'LotArea': 10000,
    'YearBuilt': 2023,
    'TotalBsmtSF': 2000,
    '1stFlrSF': 1500,
    '2ndFlrSF': 500,
    'FullBath': 2,
    'HalfBath': 1,
    'BedroomAbvGr': 3,
    'GarageCars': 2,
    'avg_price_neighborhood': 300000
}

house_price = model.predict([house_features])

# Print the predicted price
print('Predicted price:', house_price[0])
