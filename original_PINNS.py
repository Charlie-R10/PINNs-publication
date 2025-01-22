import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.numpy_ops import np_config
import os
import math
np_config.enable_numpy_behavior()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Want to review the layers and also activation function

# Code to define model architecture
def diffusion_PINN():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='tanh', input_shape=(1,))
    ])

    # Number of hidden layers
    for i in range(4):
        model.add(tf.keras.layers.Dense(64, activation='tanh'))

    # Output layer
    model.add(tf.keras.layers.Dense(1, activation='linear', name='output_layer'))

    return model


# Create an instance of model
model = diffusion_PINN()
normal_neural_net = diffusion_PINN()


# Loss function for physics informed loss
def pinns_loss():

    # Coefficients/constants
    D = 1.0  # Diffusion coefficient
    sigma_a = 0.5  # Absorption cross-section
    nu_Sigma_f = 0.2  # Fission-related term
    S = 2.0  # Source term - started with no net neutron source
    L = D / sigma_a  # L term (from book)

    x = tf.constant(np.linspace(0, a, 100), dtype=tf.float32) # Linearly space 100 points over domain

    # Compute the first and the second derivatives of phi_pred with respect to x
    with tf.GradientTape() as t1:
        t1.watch(x)
        with tf.GradientTape() as t2:
            t2.watch(x)
            phi_pred = model(x)
        dphi_dx = t2.gradient(phi_pred, x)
    d2phi_dx2 = t1.gradient(dphi_dx, x)

    # Residual loss (from book)
    L_residual = (d2phi_dx2 - (1 / L ** 2) * phi_pred)  # rest of eq:  - nu_Sigma_f * phi_pred - S

    # Boundary condition losses

    # Boundary condition J(a) = 0
    L_residual += abs(dphi_dx[-1])

    # Boundary condition phi(a) = 0
    L_residual += abs(phi_pred[-1])

    # Boundary condition providing an initial value phi(a) = 1.68
    L_residual += abs(1.68 - phi_pred[0])

    # MSE of residual
    mse_residual = tf.reduce_mean(tf.square(L_residual))

    return mse_residual


# Traditional loss (MAE same as normal neural net)
def traditional_loss(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)


# Combined loss of PINNS and NN
def combined_loss(y_true, y_pred):
    trad_loss = traditional_loss(y_true, y_pred)
    pinn_loss_val = pinns_loss()
    total_loss = trad_loss + 5*pinn_loss_val # '5' is a scalable parameter - can change this to alter loss function
    return total_loss


# Compile model
model.compile(loss=combined_loss, optimizer='adam', run_eagerly=True)
normal_neural_net.compile(loss=['MeanAbsoluteError'], optimizer='adam')

# Define parameters for the problem
D = 1.0  # Diffusion coefficient
sigma_a = 0.5  # Absorption cross-section
nu_Sigma_f = 0.2  # Fission-related term
S = 2.0  # Source term - set as 2 for now
a = 10.0  # Length of the domain
L = math.sqrt(D / sigma_a)  # L^2 = D/sigma_a

# x is number of points for data and spaced over a certain part
num_points = 4
x = np.linspace(0, 2, num_points)
x_val_points = np.linspace(0, a, 100)

def generate_training_data(x):

    phi = 4*S*(((np.sinh((a-x)/L))+((2*D)/L)*np.cosh((a-x)/L))/((((2*(D/L)+1)**2)*np.exp(a/L))-(((2*(D/L)-1)**2)*np.exp(-a/L))))
    return x, phi


# Validation data
x_train, phi_train_true = generate_training_data(x)
x_test, phi_test = generate_training_data(x_val_points)



num_epochs = 1000  # Change if necessary
batch_size = 32


# Generate data for both models
history = model.fit(x_train, phi_train_true, epochs=num_epochs, batch_size=batch_size)
history_nn = normal_neural_net.fit(x_train, phi_train_true, epochs=num_epochs, batch_size=batch_size)


# Plot analytical solution, traditional NN and PINN
plt.plot(x_test, phi_test, label='Analytical solution')
plt.plot(x_test, model.predict(x_test), label='PINN')
plt.plot(x_test, normal_neural_net.predict(x_test), label='Traditional NN')
plt.scatter(x_train, phi_train_true)
plt.legend()
plt.show()



predicted_pinn = model.predict(x_test)
predicted_nn = normal_neural_net.predict(x_test)


# Function to display r^2, RMSE, MAE, max error and max % error for model
def model_metrics(values, model):
    predicted_data = np.array(values, dtype=np.float32)
    y_true = np.array(phi_test, dtype=np.float32)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, max_error, r2_score
    print(f'r^2 ({model}) = {r2_score(y_true, predicted_data)}')
    print(f'RMSE ({model}) = {np.sqrt(mean_squared_error(y_true, predicted_data))}')
    print(f'MAE ({model}) = {mean_absolute_error(y_true, predicted_data)}')
    print(f'MAE PERCENTAGE ({model}) = {mean_absolute_percentage_error(y_true, predicted_data) * 100000}')
    print(f'MAX ERROR ({model}) = {max_error(y_true, predicted_data) * 100000}')


model_metrics(predicted_pinn, 'PINN')
model_metrics(predicted_nn, 'NN')







