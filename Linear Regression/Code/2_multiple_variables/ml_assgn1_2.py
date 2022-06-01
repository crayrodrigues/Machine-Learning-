from load_data_ex2 import *
from normalize_features import *
from gradient_descent import *
from calculate_hypothesis import *
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex2()

# Normalize
X_normalized, mean_vec, std_vec = normalize_features(X)

# After normalizing, we append a column of ones to X, as the bias term
column_of_ones = np.ones((X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X_normalized = np.append(column_of_ones, X_normalized, axis=1)

# initialise trainable parameters theta, set learning rate alpha and number of iterations
theta = np.zeros((3))
alpha = 0.2
iterations = 100

# plot predictions for every iteration?
do_plot = True

# call the gradient descent function to obtain the trained parameters theta_final
theta_final = gradient_descent(X_normalized, y, theta, alpha, iterations, do_plot)

#########################################
# Write your code here
# Create two new samples: (1650, 3) and (3000, 4)
# Calculate the hypothesis for each sample, using the trained parameters theta_final
# Make sure to apply the same preprocessing that was applied to the training data
# Print the predicted prices for the two samples

#Create a matrix with the 2 new samples.
X2 = np.matrix('1650, 3; 3000, 4')

#Normalize the samples.
X2_norm = (X2-mean_vec)/std_vec


# After normalizing, we append a column of ones to X, as the bias term
column_of_ones = np.ones((X2_norm.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X2_norm= np.append(column_of_ones, X2_norm, axis=1)

#Calculate hypothesis
y2= X2_norm * theta_final.reshape((len(theta_final),1))

for i in range(2):
    print()
    print("For x{} --> x normalized is {} and y:{}".format(X2[i], X2_norm[i], y2[i]))


########################################/

