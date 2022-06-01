import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full [:, 0] = f1
X_full[:, 1] = f2
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

X_phonemes_1_2 = np.zeros(((np.sum(phoneme_id==1)+ np.sum(phoneme_id==2)), 2))
groundTruth=np.zeros(((np.sum(phoneme_id==1)+ np.sum(phoneme_id==2)), 1))
idx=0
for i in range (len(phoneme_id)):
    if (phoneme_id[i] ==1 or phoneme_id[i]==2):
        X_phonemes_1_2[idx] = X_full[i, :]
        groundTruth[idx]=phoneme_id[i]
        idx +=1

########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"

#phoneme 1 k = 3 load pretrained data.
data01_03 = np.load('data/GMM_params_phoneme_01_k_03.npy', allow_pickle=True)
data01_03 = dict(enumerate(data01_03.flatten(), 0))

#Save means, covariances and weighted mixture of gaussians of the pretrained data of phoneme 1, k=3
mu_01_03 = data01_03[0]["mu"]
s_01_03 = data01_03[0]["s"]
p_01_03 = data01_03[0]["p"]

#Calculate how likely each data sample belongs to the k gaussians of phoneme 1.
Z_01_03= get_predictions(mu_01_03, s_01_03, p_01_03, X_phonemes_1_2)
Z_13= np.sum(Z_01_03, axis=1)

#phoneme 2 k = 3 load pretrained data.
data02_03 = np.load('data/GMM_params_phoneme_02_k_03.npy', allow_pickle=True)
data02_03 = dict(enumerate(data02_03.flatten(), 0))

#Save means, covariances and weighted mixture of gaussians of the pretrained data of phoneme 1, k=3
mu_02_03 = data02_03[0]["mu"]
s_02_03 = data02_03[0]["s"]
p_02_03 = data02_03[0]["p"]

#Calculate how likely each data sample belongs to the k gaussians of phoneme 2.
Z_02_03= get_predictions(mu_02_03, s_02_03, p_02_03, X_phonemes_1_2)
Z_23= np.sum(Z_02_03, axis=1)

#For each data sample, if the probability obtained for phoneme 1 is higher than for phoneme 2
# the value of the prediction of that data sample will be 1, otherwise 2.
prediction_03= np.zeros(((np.sum(phoneme_id==1)+ np.sum(phoneme_id==2)), 1))
for i in range(len(X_phonemes_1_2)):
    if Z_13[i] > Z_23[i]:
        prediction_03[i] = 1
    else:
        prediction_03[i] =2

#measure the accuracy of the model using the sklearn.
accuracy_03= accuracy_score(groundTruth, prediction_03, normalize=True, sample_weight=None)

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy_03*100))


##################################################
k=6
##phoneme 1 k = 6 load pretrained data.
data01_06 = np.load('data/GMM_params_phoneme_01_k_06.npy', allow_pickle=True)
data01_06 = dict(enumerate(data01_06.flatten(), 0))

#Save means, covariances and weighted mixture of gaussians of the pretrained data of phoneme 1, k=6
mu_01_06 = data01_06[0]["mu"]
s_01_06 = data01_06[0]["s"]
p_01_06 = data01_06[0]["p"]

#Calculate how likely each data sample belongs to the k gaussians of phoneme 1.
Z_01_06= get_predictions(mu_01_06, s_01_06, p_01_06, X_phonemes_1_2)
Z_16= np.sum(Z_01_06, axis=1)

#phoneme 2 k = 6 load pretrained data.
data02_06 = np.load('data/GMM_params_phoneme_02_k_06.npy', allow_pickle=True)
data02_06 = dict(enumerate(data02_06.flatten(), 0))

#Save means, covariances and weighted mixture of gaussians of the pretrained data of phoneme 2, k=6
mu_02_06 = data02_06[0]["mu"]
s_02_06 = data02_06[0]["s"]
p_02_06 = data02_06[0]["p"]

#Calculate how likely each data sample belongs to the k gaussians of phoneme 2.
Z_02_06= get_predictions(mu_02_06, s_02_06, p_02_06, X_phonemes_1_2)
Z_26= np.sum(Z_02_06, axis=1)

#For each data sample, if the probability obtained for phoneme 1 is higher than for phoneme 2
# the value of the prediction of that data sample will be 1, otherwise 2.
prediction_06= np.zeros(((np.sum(phoneme_id==1)+ np.sum(phoneme_id==2)), 1))
for i in range(len(X_phonemes_1_2)):
    if Z_16[i] > Z_26[i]:
        prediction_06[i] = 1
    else:
        prediction_06[i] =2

#Measure the accuracy of the model using the sklearn.
accuracy_06= accuracy_score(groundTruth, prediction_06, normalize=True, sample_weight=None)
print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy_06*100))
########################################/


################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()