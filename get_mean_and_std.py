import numpy as np
training_data_size = 10000
latent_space_size = 60

filename = input("filename:") #Eg : latent_space_values_T_1.0
data = open(filename).read().split('\n')
data = [[float(data[j][i] for i in range(latent_space_size)] for j in range(training_data_size)]

#Take transpose so each row is a latent space dimension
latent_space_values = [[data[j][i] for j in range(training_data_size)] for i in range(latent_space_size)]

ave = [np.average(latent_space_values[i] for i in range(latent_space_size)]
std = [np.std(latent_space_values[i] for i in range(latent_space_size)]

for i in range(latent_space_size):
    plt.figure()
    plt.hist(latent_space_values[i])
    
plt.show()