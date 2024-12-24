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

//Get the training/testing data from the specfified file
vector<DataPoint> get_data(const std::string& filename, uint32_t lattice_size)
{
	//Each line in file consists of each site value followed by
	// 1 0 if the state is ferromagnetic or
	// 0 1 if the state if paramagnetic.
	// These are the entries for the output vector
	
	std::ifstream file(filename);
	//Get the number of lines data points in the file
	uint32_t n;
	file >> n;
	
	vector<DataPoint> vec;
	vec.reserve(n);
	
	for(uint32_t i=0;i<n;i++)
	{
		VectorXf inputs(lattice_size);
		VectorXf expected_outputs(2);
		
		for(uint32_t j=0;j<lattice_size;j++) file >> inputs(j);
		file >> expected_outputs(0);
		file >> expected_outputs(1)
		
		vec.push_back({inputs, expected_outputs});	
	}
	
	return vec;
}

int main()
{
	//All of the training data and testing data has been generated using MCMC
	
	//Phase classification with 4x4 lattices
	std::cout << "4x4  lattice\n";
	std::vector<uint32_t> layer_sizes4 = {4*4,20,20,20,20,20,2};
	for(uint32_t i=0;i<9;i++)
	{
		NeuralNetwork n4(layer_sizes4);
		std::stringstream train_filename_ss;
		train_filename_ss << "training_data_4_" << i << ".txt";
		vector<DataPoint> training_data = get_data(train_filename_ss.str(),4*4);
		n4.train(training_data,1000,10,0.01);
		
		vector<DataPoint> testing_data = get_data("testing_data_4.txt",4*4);
		float error = 0.0f;
		for(DataPoint& datapoint: testing_data)
			error += (data_point.expected_outputs - 
		n4.calculate_outputs(data_point.inputs)).norm();
		std::cout << "Training set size = 10^" << i << ", Error = " << error << '\n';
	}
	
	//Phase classification with 8x8 lattices
	std::cout << "8x8 lattice\n";
	std::vector<uint32_t> layer_sizes8 = {8*8,65,65,65,65,65,2};
	for(uint32_t i=0;i<9;i++)
	{
		NeuralNetwork n8(layer_sizes8);
		std::stringstream train_filename_ss;
		train_filename_ss << "training_data_8_" << i << ".txt";
		vector<DataPoint> training_data = get_data(train_filename_ss.str(),8*8);
		n4.train(training_data,1000,10,0.01);
		
		vector<DataPoint> testing_data = get_data("testing_data_8.txt",8*8);
		float error = 0.0f;
		for(DataPoint& datapoint: testing_data)
			error += (data_point.expected_outputs - 
		n8.calculate_outputs(data_point.inputs)).norm();
		std::cout << "Training set size = 10^" << i << ", Error = " << error << '\n';
	}
	
	//Phase classification with 16x16 lattices
	std::cout << "16x16 lattice\n";
	std::vector<uint32_t> layer_sizes16 = {16*16,260,260,260,260,260,2};
	for(uint32_t i=0;i<9;i++)
	{
		NeuralNetwork n16(layer_sizes16);
		std::stringstream train_filename_ss;
		train_filename_ss << "training_data_16_" << i << ".txt";
		vector<DataPoint> training_data = get_data(train_filename_ss.str(),16*16);
		n4.train(training_data,1000,10,0.01);
		
		vector<DataPoint> testing_data = get_data("testing_data_16.txt",16*16);
		float error = 0.0f;
		for(DataPoint& datapoint: testing_data)
			error += (data_point.expected_outputs - 
		n16.calculate_outputs(data_point.inputs)).norm();
		std::cout << "Training set size = 10^" << i << ", Error = " << error << '\n';
	}
	
	//Phase classification with 32x32 lattices
	std::cout << "32x32 lattice\n";
	std::vector<uint32_t> layer_sizes32 = {32*32,1025,1025,1025,1025,1025,2};
	for(uint32_t i=0;i<9;i++)
	{
		NeuralNetwork n32(layer_sizes32);
		std::stringstream train_filename_ss;
		train_filename_ss << "training_data_32_" << i << ".txt";
		vector<DataPoint> training_data = get_data(train_filename_ss.str(),32*32);
		n4.train(training_data,1000,10,0.01);
		
		vector<DataPoint> testing_data = get_data("testing_data_32.txt",32*32);
		float error = 0.0f;
		for(DataPoint& datapoint: testing_data)
			error += (data_point.expected_outputs -
			n32.calculate_outputs(data_point.inputs)).norm();
		std::cout << "Training set size = 10^" << i << ", Error = " << error << '\n';
	}
}