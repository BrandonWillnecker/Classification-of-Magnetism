#include "Eigen/Dense"
#include "NeuralNetwork.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

using Eigen::VectorXf;
using Eigen::MatrixXf;
using std::vector

int main()
{
	//Determining the critical temperature with 32x32 lattices
	
	//Pretrained neural network that can classify 32x32 lattices
	NeuralNetwork nn("neural_network_crit_temp.txt");
	
	//The testing data is arranged into 20 groups of 100 for each temperature value
	// 0.0 < T < 2.0
	constexpr int n = 20;
	constexpr int group_size = 100;
	vector<float> f_values;
	f_values.resize(n);
	vector<DataPoint> testing_data = get_data("testing_data_crit_temp.txt",32*32);
	
	for(int i=0;i<n;i++){ // for each temperature value
		float f = 0.0f;
		for(int j=0;j<group_size;j++){ // for each sample lattice in that group_size
			//Recall : f(T) = sum_i^N(output_i(0) - output_i(1))/N
			f += nn.calculate_outputs(testing_data[i*group_size + j])[0] -
				nn.calculate_outputs(testing_data[i*group_size + j])[1]
		}
		f_values[i] = f / (float)group_size;
	}
	
	//Save f_values to a file so that that f(T) can have a sigmoid fitted to it using Python
	// or some other means
}