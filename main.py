import numpy as np
import matplotlib.pyplot as plt

# Example stock prices (daily closing prices in $1000s)
x_train = np.array([150.0, 152.5, 153.0, 151.0, 155.5, 157.0])  # Previous day's closing price (in $1000s)
y_train = np.array([152.5, 153.0, 151.0, 155.5, 157.0, 159.5])  # Next day's closing price (in $1000s)

# Feature scaling (normalize the input data)
x_mean = np.mean(x_train)
x_std = np.std(x_train)
x_train_scaled = (x_train - x_mean) / x_std

y_mean = np.mean(y_train)
y_std = np.std(y_train)
y_train_scaled = (y_train - y_mean) / y_std

# Function to calculate the cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    total_cost = (1 / (2 * m) * cost)
    return total_cost

# Function to compute the gradients
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

# Gradient descent function
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, compute_cost, compute_gradient):
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        b -= alpha * dj_db
        w -= alpha * dj_dw

        J_history.append(compute_cost(x, y, w, b))
        p_history.append([w, b])
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e}, w: {w: 0.4f}, b: {b: 0.4f}")
    return w, b, J_history, p_history

# Initialize parameters
w_init = 0
b_init = 0
iterations = 10000
alpha = 1.0e-4  # Reduced learning rate

# Run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train_scaled, y_train_scaled, w_init, b_init, alpha, iterations, compute_cost, compute_gradient)

print(f"(w,b) found by gradient descent: ({w_final},{b_final})")

# Plot cost vs iteration
plt.plot(J_hist)
plt.title('Cost vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# Predictions for future stock prices
# To make predictions, scale the input and reverse scale the output
def predict_price(x, w, b, x_mean, x_std, y_mean, y_std):
    x_scaled = (x - x_mean) / x_std
    y_scaled = w * x_scaled + b
    return y_scaled * y_std + y_mean

print(f"Prediction for next day price after $1500 stock: {predict_price(150.0, w_final, b_final, x_mean, x_std, y_mean, y_std):.2f} Thousand dollars")
print(f"Prediction for next day price after $1600 stock: {predict_price(160.0, w_final, b_final, x_mean, x_std, y_mean, y_std):.2f} Thousand dollars")
print(f"Prediction for next day price after $2000 stock: {predict_price(200.0, w_final, b_final, x_mean, x_std, y_mean, y_std):.2f} Thousand dollars")
