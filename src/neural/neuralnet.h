#ifndef NEURALNET_H
#define NEURALNET_H

#include <armadillo>

class NeuralNet
{
private:
	static constexpr int	InputCount = 10;						// The maximum number of characters to process.
	static constexpr int	OutputCount = (2 * InputCount);			// What if every character is separated by a space?
	static constexpr int	HiddenNeuronCount = 24;					// currently arbitrary (# per layer)
	static constexpr int	HiddenLayerCount = 3;					// currently arbitrary
	static constexpr int	LayerCount = (2 + HiddenLayerCount);	// only somewhat arbitrary
	static constexpr int	BiasCount = LayerCount - 1;
	static constexpr int	WeightCount = LayerCount - 1;

	arma::mat				layers[LayerCount];						// neuron count x 1
	arma::mat				biases[BiasCount];						// neuron count x 1
	arma::mat				weights[WeightCount];					// previous layer's neuron count * next layer's neuron count

public:
	void					Train(std::wstring inputs);
	std::vector<int>		Run(std::vector<int> inputs);		// The trigger.

	NeuralNet();
};

#endif