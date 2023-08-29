#ifndef NEURALNET_H
#define NEURALNET_H

#include <armadillo>

class NeuralNet
{
public:
	static constexpr double	LearningRate = 10.0;

	static constexpr int	InputCount = 100;												// The maximum number of characters to process.
	static constexpr int	OutputCount = InputCount;
	// static constexpr int	OutputCount = (2 * InputCount);									// What if every character is separated by a space?
	static constexpr int	HiddenNeuronCount = InputCount * 2;								// currently arbitrary (# per layer)
	static constexpr int	HiddenLayerCount = 4;											// currently arbitrary
	static constexpr int	LayerCount = (2 + HiddenLayerCount);							// only somewhat arbitrary
	static constexpr int	LayerDerivativeCount = LayerCount - 1;							// The first layer doesn't need any derivatives.
	static constexpr int	BiasCount = LayerCount - 1;
	static constexpr int	WeightCount = LayerCount - 1;

private:
	arma::mat				layerInputs[LayerCount - 1];									// neuron count x 1
	arma::mat				layerOutputs[LayerCount];										// neuron count x 1
	arma::mat				activationDerivatives[LayerDerivativeCount];
	arma::mat				outputDerivatives[LayerDerivativeCount];
	arma::mat				biases[BiasCount];												// neuron count x 1
	arma::mat				weights[WeightCount];											// previous layer's neuron count * next layer's neuron count
	arma::mat				weightDerivatives[WeightCount];
	arma::mat				biasDerivatives[BiasCount];

public:
	std::vector<double>		Forward(std::vector<double> inputs, bool print);				// Forward propagation
	void					Back(std::vector<double> costs);								// Back propagation
	void					Train(std::vector<double> inputs, std::vector<double> targets, bool print);	// Training!
	void					ClearInOutputs();												// Duh

	NeuralNet();
};

#endif