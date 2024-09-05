#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <string>
#include "NeuralNetwork.h"

int main()
{
	//Load Neural Network from save file
	NeuralNetwork::NeuralNetwork nn("XY_NN_T_1.0.txt");
	
	//Load the latent space averages and standard deviations from save file
	std::ifstream ave_file("ave_T_1.0.txt");
	std::ifstream std_file("std_T_1.0.txt");
	
	std::vector<float> aves(60);
	std::vector<float> stds(60);
	
	for(int i=0;i<60;i++){
		ave_file >> aves[i];
		std_file >> stds[i];
	}

	//Set up a normal distribution generator
	std::random_device rd{};
    std::mt19937 gen{rd()};
	
	//Create a random latent space vector from this distribution
	std::vector<float> latent_space_vector(60);
	for(int i=0;i<60;i++){
		std::normal_distribution dist{aves[i], stds[i]};
		latent_space_vector[i] = dist(gen);
	}
    
	
	//We can then sample the A(x,y) field corresponding to this vector
	std::vector<float> A_field = nn.outputs_from_layer(5,latent_space_vector);
	
	//This can be repeated as many times as needed to get a collection of A(x,y) fields
}