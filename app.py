import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 Simple Linear Regression Project")
st.markdown("---")

# SECTION 1: Least Squares Method
st.header("1️⃣ Linear Regression using Least Squares")

x_ls = np.array([2, 3, 5, 7, 9])
y_ls = np.array([1, 2, 4, 5, 6])
n_ls = len(x_ls)

sum_x = np.sum(x_ls)
sum_y = np.sum(y_ls)
sum_xy = np.sum(x_ls * y_ls)
sum_x_squared = np.sum(x_ls ** 2)

m_ls = (n_ls * sum_xy - sum_x * sum_y) / (n_ls * sum_x_squared - sum_x ** 2)
c_ls = (sum_y - m_ls * sum_x) / n_ls

st.write(f"**Slope (m):** {m_ls:.4f}")
st.write(f"**Intercept (c):** {c_ls:.4f}")

fig1, ax1 = plt.subplots()
ax1.scatter(x_ls, y_ls, color='blue', label='Data Points')
ax1.plot(x_ls, m_ls * x_ls + c_ls, color='red', label=f'y = {m_ls:.2f}x + {c_ls:.2f}')
ax1.set_title("Least Squares Regression")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend()
st.pyplot(fig1)

st.markdown("---")

# SECTION 2: Gradient Descent Method
st.header("2️⃣ Linear Regression using Gradient Descent")

x_gd = np.array([1, 2, 3, 4, 5])
y_gd = np.array([2, 3, 5, 7, 8])
n_gd = len(x_gd)

learning_rate = 0.05
epochs = 200
m = 0
c = 0

for epoch in range(epochs):
    y_pred = m * x_gd + c
    dm = (-2 / n_gd) * np.sum(x_gd * (y_gd - y_pred))
    dc = (-2 / n_gd) * np.sum(y_gd - y_pred)
    m -= learning_rate * dm
    c -= learning_rate * dc
    if (epoch + 1) % 20 == 0:
        st.write(f"Epoch {epoch+1}: m = {m:.4f}, c = {c:.4f}")

fig2, ax2 = plt.subplots()
ax2.scatter(x_gd, y_gd, color='green', label='Data Points')
ax2.plot(x_gd, m * x_gd + c, color='orange', label=f'y = {m:.2f}x + {c:.2f}')
ax2.set_title("Gradient Descent Regression")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend()
st.pyplot(fig2)

st.markdown("---")

# SECTION 3: Error Metrics
st.header("3️⃣ Error Metrics")

def calculate_metrics(y_actual, y_pred, num_predictors):
    n = len(y_actual)
    residuals = y_actual - y_pred
    MAE = np.mean(np.abs(residuals))
    SSE = np.sum(residuals ** 2)
    MSE = np.mean(residuals ** 2)
    RMSE = np.sqrt(MSE)
    MAPE = np.mean(np.abs(residuals / y_actual)) * 100
    SS_total = np.sum((y_actual - np.mean(y_actual)) ** 2)
    R_square = 1 - (SSE / SS_total)
    Adjusted_R_square = 1 - ((SSE / (n - num_predictors - 1)) / (SS_total / (n - 1)))

    return MAE, SSE, MSE, RMSE, MAPE, R_square, Adjusted_R_square

y_actual = np.array([3, 4, 5, 6])
y_pred = np.array([2.5, 4.2, 4.8, 6.3])
num_predictors = 1

MAE, SSE, MSE, RMSE, MAPE, R2, R2_adj = calculate_metrics(y_actual, y_pred, num_predictors)

st.write(f"**Mean Absolute Error (MAE):** {MAE:.4f}")
st.write(f"**Sum of Squared Errors (SSE):** {SSE:.4f}")
st.write(f"**Mean Squared Error (MSE):** {MSE:.4f}")
st.write(f"**Root Mean Squared Error (RMSE):** {RMSE:.4f}")
st.write(f"**Mean Absolute Percentage Error (MAPE):** {MAPE:.2f}%")
st.write(f"**R-squared:** {R2:.4f}")
st.write(f"**Adjusted R-squared:** {R2_adj:.4f}")
