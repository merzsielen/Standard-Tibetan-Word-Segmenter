#include "neuralnet.h"

#include <cstdlib>
#include <time.h>

double Sigmoid(double x)
{
	return x / (1 + abs(x));
}

void NeuralNet::Train(std::wstring inputs)
{

}

std::vector<int> NeuralNet::Run(std::vector<int> inputs)
{
	// Let's clear all the values first.
	for (int i = 0; i < LayerCount; i++) layers[i].zeros();

	// Now, let's feed in our inputs.
	for (int i = 0; i < layers[0].n_rows; i++)
	{
		layers[0](0, i) = inputs[i];
	}

	// And now we iteratively propagate them forward.
	for (int i = 0; i < WeightCount; i++)
	{
		layers[i + 1] = layers[i] * weights[i];
		
		if (i + 1 == LayerCount - 1) break;

		for (int j = 0; j < layers[i + 1].n_cols; j++)
		{
			layers[i + 1](0, j) += biases[i](0, j);
			layers[i + 1](0, j) = Sigmoid(layers[i + 1](0, j));
		}
	}

	std::vector<int> outputs;
	for (int i = 0; i < layers[LayerCount - 1].n_cols; i++) outputs.push_back((int)layers[LayerCount - 1](0, i));
	return outputs;
}

NeuralNet::NeuralNet()
{
	srand(time(NULL));

	/*
		First, let's instantiate our layers.

		We need to do our inputs and outputs first.
	*/
	layers[0] = arma::mat(1, InputCount);
	layers[LayerCount - 1] = arma::mat(1, OutputCount);
	
	// Now, we go through and instantiate each layer.
	for (int i = 1; i < LayerCount - 1; i++)
	{
		layers[i] = arma::mat(1, HiddenNeuronCount);
	}

	// Next, we're going to instantiate our biases.
	biases[0] = arma::mat(1, HiddenLayerCount);

	for (int i = 1; i < LayerCount - 1; i++)
	{
		biases[i] = arma::mat(1, HiddenNeuronCount);
	}

	// And now we're free to go through and instantiate
	// and randomize the weights.
	for (int i = 0; i < HiddenLayerCount; i++)
	{
		weights[i] = arma::mat(layers[i].n_cols, layers[i + 1].n_cols, arma::fill::randn);
	}
}