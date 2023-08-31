#include "neuralnet.h"

#include <cstdlib>
#include <time.h>

double Logistic(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double LogisticDerivative(double x)
{
	double fx = Logistic(x);
	if (fx == 0) return 0.5;
	return fx - (fx * fx);
}

double OutputError(double target, double output)
{
	double diff = target - output;
	return 0.5 * (diff * diff);
}

double Clamp(double input)
{
	if (input < 0.00001) return 0.0;
	if (input > 0.99999) return 1.0;
	return input;
}

std::vector<double> NeuralNet::Forward(std::vector<double> inputs)
{
	// First, let's feed in our inputs.
	for (int i = 0; i < layerOutputs[0].n_cols; i++)
	{
		layerOutputs[0](0, i) = inputs[i];
	}

	// And now we iteratively propagate them forward.
	for (int i = 0; i < WeightCount; i++)
	{
		layerInputs[i] = layerOutputs[i] * weights[i];

		// Now we apply the activation function to the inputs
		// to get the outputs. Separating inputs and outputs
		// makes backward propagation easier.
		for (int j = 0; j < layerInputs[i].n_cols; j++)
		{
			layerInputs[i](0, j) += biases[i](0, j);
			layerOutputs[i + 1](0, j) = Clamp(Logistic(layerInputs[i](0, j)));
		}
	}

	// Penultimately, we pull out all our outputs.
	std::vector<double> outputs;
	for (int i = 0; i < layerOutputs[LayerCount - 1].n_cols; i++) outputs.push_back(layerOutputs[LayerCount - 1](0, i));

	// Job done.
	return outputs;
}

void NeuralNet::Back(std::vector<double> costs)
{
	/*
		First, let's compute the derivatives of the
		error function with respect to the output of
		each output neuron.
	*/

	for (int i = 0; i < layerOutputs[LayerCount - 1].n_cols; i++)
	{
		/*
			For each output neuron, the derivative of the
			error function with respect to the output is
			just the difference between the output and
			the target.
		*/
		outputDerivatives[LayerDerivativeCount - 1](0, i) = costs[i];
		double z = layerInputs[LayerCount - 2](0, i);
		activationDerivatives[LayerDerivativeCount - 1](0, i) = LogisticDerivative(z);
	}

	for (int i = WeightCount - 1; i > -1; i--)
	{
		/*
			For back propagation, we start at the end.

			We must work backwards, cycling from weights to
			neurons to biases and around again. Thus, we'll start
			with weights.

			The derivative of the error with respect to a weight (which is
			between neuron OUT and neuron IN) is the derivative of the error
			with respect to IN's output times the derivative of IN's output
			with respect to the weight. This is defined as the
			LogisticDerivative() of IN's sum inputs times the derivative of IN's
			sum inputs with regard to the weight (which is the the output of OUT).

			We'll go along each row in each weight matrix since these all
			involve the same succeeding neuron.
		*/
		for (int j = 0; j < weights[i].n_rows; j++)
		{
			for (int k = 0; k < weights[i].n_cols; k++)
			{
				double neuroDeriv = outputDerivatives[i](0, k);
				double activaDeriv = activationDerivatives[i](0, k);
				double output = layerOutputs[i](0, j);

				weightDerivatives[i](j, k) = neuroDeriv * activaDeriv * output;
			}
		}

		/*
			The derivative of the error with respect to a particular bias is the
			derivative of the error with respect to the neuron's output times the
			derivative of the neuron's output with respect to the bias. This is
			the LogisticDerivative() of the neuron (times the derivative of the
			sum-input of the neuron with respect to the bias, which will be 1).
		*/
		for (int j = 0; j < biases[i].n_cols; j++)
		{
			double neuroDeriv = outputDerivatives[i](0, j);
			double activaDeriv = activationDerivatives[i](0, j);

			biasDerivatives[i](0, j) = neuroDeriv * activaDeriv;
		}

		// We don't need to calculate the rest for the first layer.
		if (i == 0) break;

		/*
			The derivative of the error with respect to a neuron's output
			is the sum of the derivatives of the error with respect to the
			outputs of the neurons in the next layer times the derivatives of
			these neurons' outputs with respect to the output of the neuron in
			question.

			The derivative of a neuron's output (which we'll call neuron OUT) with
			respect to the output of a preceding neuron (which will be neuron IN)
			is the derivative of the logistic function with regard to the output of
			OUT times the derivative of the output of OUT with respect to the output
			of IN (which is the weight associated with the connection between IN and
			OUT).

			In other words: what we'll just be shortening to the "derivative" of
			a neuron is the sum of the "derivatives" of the neurons in the next
			layer times the derivatives of their outputs with respect to the
			neuron in question, which is just the LogisticDerivative() of the
			next-layer neuron times the weight between the two neurons.
		*/
		for (int j = 0; j < layerOutputs[i].n_cols; j++)
		{
			double sum = 0.0;

			for (int k = 0; k < weights[i].n_cols; k++)
			{
				double neuroDeriv = outputDerivatives[i](0, k);
				double activaDeriv = activationDerivatives[i](0, k);
				double weight = weights[i](j, k);

				sum += neuroDeriv * activaDeriv * weight;
			}

			outputDerivatives[i - 1](0, j) = sum;
			activationDerivatives[i - 1](0, j) = LogisticDerivative(layerInputs[i - 1](0, j));
		}
	}

	for (int i = 0; i < WeightCount; i++)
	{
		// Update the weights.
		for (int j = 0; j < weights[i].n_rows; j++)
		{
			for (int k = 0; k < weights[i].n_cols; k++)
			{
				weights[i](j, k) -= LearningRate * weightDerivatives[i](j, k);
			}
		}

		// Update the biases.
		for (int j = 0; j < biases[i].n_cols; j++)
		{
			biases[i](0, j) -= LearningRate * biasDerivatives[i](0, j);
		}
	}

	// Clear
	for (int i = 0; i < LayerDerivativeCount; i++)
	{
		activationDerivatives[i].zeros();
		outputDerivatives[i].zeros();
	}
	for (int i = 0; i < WeightCount; i++) weightDerivatives[i].zeros();
	for (int i = 0; i < BiasCount; i++) biasDerivatives[i].zeros();
}

void NeuralNet::ClearInOutputs()
{
	for (int i = 0; i < LayerCount; i++)
	{
		if (i < LayerCount - 2)
		{
			layerInputs[i].zeros();
		}

		layerOutputs[i].zeros();
	}
}

double NeuralNet::Train(std::vector<double> inputs, std::vector<double> targets)
{
	std::vector<double> output = Forward(inputs);

	double loss = 0.0;
	for (int i = 0; i < output.size(); i++)
	{
		output[i] -= targets[i];
		loss += output[i] * output[i];
	}

	Back(output);

	ClearInOutputs();

	return loss;
}

NeuralNet::NeuralNet()
{
	// Seed arma random
	arma::arma_rng::set_seed_random();

	/*
		First, let's instantiate our layers.

		We need to do our inputs and outputs first.
	*/
	layerOutputs[0] = arma::mat(1, InputCount);

	layerInputs[LayerCount - 2] = arma::mat(1, OutputCount);
	layerOutputs[LayerCount - 1] = arma::mat(1, OutputCount);

	activationDerivatives[LayerDerivativeCount - 1] = arma::mat(1, OutputCount);
	outputDerivatives[LayerDerivativeCount - 1] = arma::mat(1, OutputCount);
	
	// Now, we go through and instantiate each layer input.
	for (int i = 1; i < LayerCount - 1; i++)
	{
		layerInputs[i - 1] = arma::mat(1, HiddenNeuronCount);
		layerOutputs[i] = arma::mat(1, HiddenNeuronCount);
	}

	// And we need to activate these derivatives.
	for (int i = 0; i < LayerDerivativeCount - 1; i++)
	{
		activationDerivatives[i] = arma::mat(1, HiddenNeuronCount);
		outputDerivatives[i] = arma::mat(1, HiddenNeuronCount);
	}

	// Next, we're going to instantiate our biases.
	for (int i = 0; i < BiasCount; i++)
	{
		biases[i] = arma::mat(1, HiddenNeuronCount, arma::fill::randu);
		biasDerivatives[i] = arma::mat(1, HiddenNeuronCount);
	}

	// And we also need to instantiate the output biases.
	biases[BiasCount - 1] = arma::mat(1, OutputCount, arma::fill::randu);
	biasDerivatives[BiasCount - 1] = arma::mat(1, OutputCount);

	// And now we're free to go through and instantiate
	// and randomize the weights.
	for (int i = 0; i < WeightCount; i++)
	{
		weights[i] = arma::mat(layerOutputs[i].n_cols, layerInputs[i].n_cols, arma::fill::randu);
		weightDerivatives[i] = arma::mat(layerOutputs[i].n_cols, layerInputs[i].n_cols);

		for (int j = 0; j < weights[i].n_rows; j++)
		{
			for (int k = 0; k < weights[i].n_cols; k++)
			{
				weights[i](j, k) *= 0.01;
			}
		}
	}
}