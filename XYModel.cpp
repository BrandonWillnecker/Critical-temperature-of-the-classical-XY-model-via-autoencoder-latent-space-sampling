#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <string>
#include "NeuralNetwork.h"

constexpr int N = 30;
constexpr int SIZE = N * N;
constexpr int MCMCSTEPS = 1000000;
constexpr float TWO_PI = 6.28318530718f;

std::vector<float> generate_sample(float T)
{
	//random grid init
	std::vector<float> grid(SIZE);
	for (int i = 0; i < SIZE; i++)
		grid[i] = TWO_PI * (float)rand()/RAND_MAX;

	for (int step = 0; step < MCMCSTEPS; step++) 
	{
		for (int col = 0; col < N; col++) {
			for (int row = 0; row < N; row++) {
				float delta = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
				int row_plus = (row + 1) % N; int row_minus = (row - 1 + N) % N;
				int col_plus = (col + 1) % N; int col_minus = (col - 1 + N) % N;

				//Only energy of neighbourhood changes
				// X' = X + delta
				//      *                   *
				//      |			        |
				// * -- X -- *   ==>   * -- X' -- *
				//      |			        |
				//      *			        *

				float theta = grid[row * N + col];
				float theta_new = theta + delta;

				//dE = E_new - E_prev
				float dE = -cosf(theta_new - grid[row_plus * N + col]) + cosf(theta - grid[row_plus * N + col])
						   -cosf(theta_new - grid[row_minus * N + col])+ cosf(theta - grid[row_minus * N + col])
				           -cosf(theta_new - grid[row * N + col_plus]) + cosf(theta - grid[row * N + col_plus])
				           -cosf(theta_new - grid[row * N + col_minus])+ cosf(theta- grid[row * N + col_minus]);

				float u = (float)rand() / RAND_MAX;

				if (dE < 0.0f || u < exp(-dE / T)) {
					//Keep Changes
					grid[row * N + col] = theta_new;
				}
				else {
					//Reject chnages
				}
			}
		}
	}

	for (int i = 0; i < SIZE; i++) {
		while (grid[i]<0.0f || grid[i]>TWO_PI) {
			if (grid[i] < 0.0f) grid[i] += TWO_PI;
			if (grid[i] > TWO_PI) grid[i] -= TWO_PI;
		}
	}

	return grid;
}

static void plot_sample(const std::vector<float>& sample)
{
	std::ofstream file("plot_theta_field.py");
	file << "import matplotlib.pyplot as plt\n";
	file << "import numpy as np\n";
	file << "field = [";
	for (int row = 0; row < N; row++) {
		file << "[";
		for (int col = 0; col < N; col++) {
			file << sample[row * N + col] << ", ";
		}
		file << "], ";
	}
	file << "]\n";
	file << "x = np.arange(0," << N << ")\n";
	file << "X,Y = np.meshgrid(x,x)\n";
	file << "U = [";
	for (int row = 0; row < N; row++) {
		file << "[";
		for (int col = 0; col < N; col++) {
			file << cosf(sample[row * N + col]) << ", ";
		}
		file << "], ";
	}
	file << "]\n";
	file << "V = [";
	for (int row = 0; row < N; row++) {
		file << "[";
		for (int col = 0; col < N; col++) {
			file << sinf(sample[row * N + col]) << ", ";
		}
		file << "], ";
	}
	file << "]\n";
	file << "plt.imshow(field,cmap='hsv',vmin=0.0,vmax=" << TWO_PI << ")\n";
	file << "plt.quiver(X,Y,U,V)\n";
	file << "plt.show()";
	file.close();
	system("plot_theta_field.py");
}

static void generate_sample_file(float T_min, float T_max, int T_steps, int nsamples){
	std::cout << "generating file of samples\n";
	float dT = (T_max - T_min) / T_steps;
	std::ofstream file("samples_file.txt");
	float T = T_min;
	while (T < T_max) {
		for (int sample_index = 0; sample_index < nsamples; sample_index++){
			std::vector<float> sample = generate_sample(T);
			for (float s : sample) file << s << ',';
			file << '\n';
		}
		T += dT;
	}
	file.close();
}

static std::vector<float> get_A(const std::vector<float>& field)
{
	//Gaussian Kernel for Auxillary field convolution
	constexpr float gauss_ker[3][3] = { {1.0f/16.0f, 1.0f/8.0f, 1.0f/16.0f},
										{1.0f/8.0f , 1.0f/4.0f, 1.0f/8.0f },
										{1.0f/16.0f, 1.0f/8.0f, 1.0f/16.0f}};
	std::vector<float> A_field(SIZE);

	for (int y = 0; y < N; y++) { //row
		for (int x = 0; x < N; x++) {//col
			float A = 0.0f;
			for (int u = 0; u < 3; u++) {//row
				for (int v = 0; v < 3; v++) {//col
					int row = (y - u - 1 + N) % N; int col = (x - v - 1 + N) % N;
					A += gauss_ker[u][v] * (1.0f - cosf(field[y * N + x] - field[row* N + col]));
				}
			}
			A_field[y * N + x] = A;
		}
	}

	return A_field;
}

static void plot_A(const std::vector<float>& A_field)
{
	std::ofstream file("plot_A_field.py");
	file << "import matplotlib.pyplot as plt\n";
	file << "import numpy as np\n";
	file << "field = [";
	for (int row = 0; row < N; row++) {
		file << "[";
		for (int col = 0; col < N; col++) {
			file << A_field[row * N + col] << ", ";
		}
		file << "], ";
	}
	file << "]\n";
	file << "plt.imshow(field,cmap='plasma',vmin=0.0,vmax=1.0)\n";
	file << "plt.show()";
	file.close();
	system("plot_A_field.py");
}

static float ave(const std::vector<float>& vec) {
	float s = 0.0f;
	for (float v : vec) s += v;
	return s / vec.size();
}

int main()
{
	srand(time(NULL));

	std::vector<size_t> LAYER_SIZES = { 225,169,127,95,71,60,71,95,127,225 };
	constexpr int EPOCHS = 5000;
	constexpr int BATCH_SIZE = 100;
	constexpr float LEARNING_RATE = 0.01f;
	constexpr int TRAINING_DATA_SIZE = 10000;

	constexpr float T_MIN = 0.001f;
	constexpr float T_MAX = 2.0f;
	constexpr int T_STEPS = 100;
	constexpr float dT = (T_MAX - T_MIN) / T_STEPS;

	//Save training data to file as a backup
	generate_sample_file(T_MIN, T_MAX, T_STEPS,TRAINING_DATA_SIZE);

	std::ifstream training_data_file("samples_file.txt");
	for (float T = T_MIN; T <= T_MAX; T += dT) {
		std::vector<std::vector<float>> training_data(TRAINING_DATA_SIZE);
		for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
			training_data.emplace_back(SIZE);
			for (int j = 0; j < SIZE; j++) training_data_file >> training_data[i][j];
		}
		std::string nn_filename = "XY_NN_T_" + std::to_string(T) + ".txt";
		NeuralNetwork::NeuralNetwork nn(nn_filename, LAYER_SIZES);
		//Inputs = training_data
		//Expected outputs = training_data
		//Keep default activation functions
		nn.train(training_data, training_data, EPOCHS, BATCH_SIZE, LEARNING_RATE);
		nn.cost_function = NeuralNetwork::COSTFUNCTION::MSE;
		
		std::ofstream latent_space_file("latent_space_values_T_" + std::to_string(T) + ".txt");
		for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
			nn.outputs(training_data[i]);
			std::vector<float> latent_space = nn.layers(5);
			for (int j = 0; j < latent_space.size(); j++) latent_space_file << latent_space[j] << ", ";
			latent_space_file << '\n';
		}
		latent_space_file.close();
	}

}