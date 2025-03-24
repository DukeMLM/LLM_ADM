from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import randint

from train import data_preprocessing


G_train, s_train, G_test, s_test, G_val, s_val = data_preprocessing(100)

# Define parameter distributions
param_dist = {
    'n_neighbors': randint(2, 50),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree']
}

# Create the KNeighborsRegressor
model = KNeighborsRegressor()

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(model, param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error')

# Train the model
random_search.fit(G_train, s_train)

# Get the best model and predictions
best_model = random_search.best_estimator_
y_pred = best_model.predict(G_test)

# Evaluate the best model's performance
mse = mean_squared_error(s_test, y_pred)
print("Best MSE:", -mse)  # Use negative MSE since scoring uses neg_mean_squared_error

# Print the best hyperparameters
print("Best hyperparameters:", random_search.best_params_)
