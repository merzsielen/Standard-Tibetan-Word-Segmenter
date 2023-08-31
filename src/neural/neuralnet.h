#ifndef NEURALNET_H
#define NEURALNET_H

#include <armadillo>

enum class LayerType { input, output, hidden, convolutional };

struct Layer
{
	LayerType				layerType;

	arma::mat				inputs;
	arma::mat				outputs;

	arma::mat				inputDerivatives;
	arma::mat				outputDerivatives;

	arma::mat				biases;
	arma::mat				weights;

	arma::mat				biasDerivatives;
	arma::mat				weightDerivatives;

	// Adding a tag to remind myself to change this when we
	// change the kernel size: [KERNEL_CHANGE].
	static constexpr int	KernelSize = 3;
	arma::mat				filter;
	arma::mat				filterDerivatives;
};

class NeuralNet
{
public:
	static constexpr double	LearningRate = 0.05;

	static constexpr int	InputCount = 784;												
	static constexpr int	OutputCount = 10;
	static constexpr int	HiddenNeuronCount = 32;											// currently arbitrary (# per layer)
	static constexpr int	HiddenLayerCount = 2;											// currently arbitrary
	static constexpr int	LayerCount = (2 + HiddenLayerCount);							// only somewhat arbitrary

private:
	Layer					layers[LayerCount];

public:
	std::vector<double>		Forward(std::vector<double> inputs);							// Forward propagation
	void					Back(std::vector<double> costs);								// Back propagation
	double					Train(std::vector<double> inputs, std::vector<double> targets);	// Training!
	void					ClearInOutputs();												// Duh

	NeuralNet(std::vector<LayerType> hiddenLayerTypes);
};

#endif