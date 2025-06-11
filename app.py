

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Given data points
x = np.array([2, 3, 5, 7, 9])
y = np.array([1, 2, 4, 5, 6])

# Number of data points
n = len(x)

# Calculate sums needed for least squares method
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x_squared = np.sum(x ** 2)

# Calculate slope (m) and intercept (c)
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
c = (sum_y - m * sum_x) / n

# Print the calculated values of m and c
print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")

# Plotting the data points
plt.scatter(x, y, color='blue', label='Data Points')

# Plotting the regression line
plt.plot(x, m * x + c, color='red', label=f'Regression Line: y = {m:.2f}x + {c:.2f}')

# Labeling the axes
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Linear Regression using Least Squares Method')

# Adding a legend
plt.legend()

# Display the plot
plt.show()
print("Slope (m):", 0.7134146341463414)

print("Intercept (c):", -0.10975609756097526)


#Compute Parmeters using Gradient Descent
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Given data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 8])

# Number of data points
n = len(x)

# Hyperparameters
learning_rate = 0.05
epochs = 200  # Number of iterations

# Initialize m (slope) and c (intercept) to zero
m = 0
c = 0

# Store history of m and c for plotting
m_history = [m]
c_history = [c]

# Gradient Descent Function
for epoch in range(epochs):
    y_pred = m * x + c  # Predicted values
    
    # Calculate the gradients
    dm = (-2/n) * np.sum(x * (y - y_pred))
    dc = (-2/n) * np.sum(y - y_pred)
    
    # Update parameters
    m -= learning_rate * dm
    c -= learning_rate * dc
    
    # Store values for plotting
    m_history.append(m)
    c_history.append(c)
    
    # Print values every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}: m = {m:.4f}, c = {c:.4f}")

# Plotting the data points
plt.scatter(x, y, color='blue', label='Data Points')

# Plotting the regression line
plt.plot(x, m * x + c, color='red', label=f'Regression Line: y = {m:.2f}x + {c:.2f}')

# Labeling the axes
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Linear Regression using Least Squares Method')

# Adding a legend
plt.legend()

# Display the plot
plt.show()
print("Epoch 20: m = 1.5555, c = 0.3606")
print("Epoch 40: m = 1.5684, c = 0.3142")
print("Epoch 60: m = 1.5775, c = 0.2812")
print("Epoch 80: m = 1.5840, c = 0.2577")
print("Epoch 100: m = 1.5886, c = 0.2411")
print("Epoch 120: m = 1.5919, c = 0.2292")
print("Epoch 140: m = 1.5943, c = 0.2208")
print("Epoch 160: m = 1.5959, c = 0.2148")
print("Epoch 180: m = 1.5971, c = 0.2105")
print("Epoch 200: m = 1.5979, c = 0.2075")


#Error Metrics
import numpy as np

# Define a function to calculate all metrics
def calculate_metrics(y_actual, y_pred, num_predictors):
    n = len(y_actual)
    
    # Mean of actual values
    y_mean = np.mean(y_actual)
    
    # Errors
    residuals = y_actual - y_pred
    
    # Metrics
    MAE = np.mean(np.abs(residuals))
    SSE = np.sum(residuals**2)
    MSE = np.mean(residuals**2)
    RMSE = np.sqrt(MSE)
    MAPE = np.mean(np.abs(residuals / y_actual)) * 100
    
    # R-squared
    SS_total = np.sum((y_actual - y_mean)**2)
    R_square = 1 - (SSE / SS_total)
    
    # Adjusted R-squared
    Adjusted_R_square = 1 - ((SSE / (n - num_predictors - 1)) / (SS_total / (n - 1)))
    
    # Print all metrics
    print(f"Mean Absolute Error (MAE): {MAE}")
    print(f"Sum of Squared Errors (SSE): {SSE}")
    print(f"Mean Squared Error (MSE): {MSE}")
    print(f"Root Mean Squared Error (RMSE): {RMSE}")
    print(f"Mean Absolute Percentage Error (MAPE): {MAPE}%")
    print(f"R-squared: {R_square}")
    print(f"Adjusted R-squared: {Adjusted_R_square}")

# Example usage:
y_actual = np.array([3, 4, 5, 6])
y_pred = np.array([2.5, 4.2, 4.8, 6.3])

# Assuming 1 predictor (simple linear regression)
num_predictors = 1

calculate_metrics(y_actual, y_pred, num_predictors)
print("Mean Absolute Error (MAE): 0.30000000000000004")
print("Sum of Squared Errors (SSE): 0.4200000000000001")
print("Mean Squared Error (MSE): 0.10500000000000002")
print("Root Mean Squared Error (RMSE): 0.32403703492039304")
print("Mean Absolute Percentage Error (MAPE): 7.666666666666668%")
print("R-squared: 0.9159999999999999")
print("Adjusted R-squared: 0.874")
