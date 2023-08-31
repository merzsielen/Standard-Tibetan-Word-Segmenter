/*
	main.cpp
*/

#include <chrono>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "csv.hpp"

#include "utility/filehandling.h"
#include "utility/scrubber.h"
#include "neural/datahandler.h"

#define VERSION 0.01

#define NONTOKENPATH "datasets/input/nontokenized.txt"
#define TOKENPATH "datasets/input/tokenized.txt"

void main()
{
	std::locale::global(std::locale("en_US.UTF-8"));
	std::wcout.imbue(std::locale());
	/*
		Just for kicks, we'll spit out the version number
		and maybe some other details later.
	*/
	std::ostringstream o;
	o << std::setprecision(2) << std::noshowpoint << VERSION;
	std::string ostr = o.str();
	std::wstring ver(ostr.begin(), ostr.end());
	std::wcout << L"Tashi delek!\n"
		<<"You are running the Lhasa Tibetan NNSegmenter.\n"
		<< L"Version: " + ver + L".\n\n";

	/*
		First, we need to clean up the input and output data.
		This involves dropping everything that isn't a sentence
		composed of standard Tibetan words. Specifically, we drop any
		non-Tibetan stuff, such as anything in the Latin alphabet
		or Chinese characters. Then, we go through and parse it into
		sentences. This is easy for our tokenized data as it is all
		formatted nice and neatly, but the wiki dump has all sorts of
		extra stuff we need to get rid of.
	*/
	/*std::wstring nontoken = Scrub(WReadFile(NONTOKENPATH));
	std::wstring token = Scrub(WReadFile(TOKENPATH));

	std::wcout << "The nontokenized corpus contains " << nontoken.size() << " characters.\n";
	std::wcout << "And the tokenized corpus contains " << token.size() << " characters.\n\n";

	WriteFile("datasets/output/scrubbed_nontokenized.txt", nontoken);
	WriteFile("datasets/output/scrubbed_tokenized.txt", token);*/

	/*
		Now, we need to go through and start preparing both
		corpora, transferring them into data structures intelligible
		to the neural network.
	*/

	/*NeuralNet* neuralNet = new NeuralNet();
	DataHandler* dataHandler = new DataHandler(neuralNet, nontoken, token);

	dataHandler->Train();

	delete neuralNet;
	delete dataHandler;*/

	NeuralNet* neuralNet = new NeuralNet();


	int maxEpochs = 20;

	for (int e = 0; e < maxEpochs; e++)
	{
		csv::CSVReader trainData("datasets/input/mnist_train.csv");

		for (csv::CSVRow& row : trainData)
		{
			std::vector<double> targets;
			for (int i = 0; i < 10; i++) targets.push_back(0.0);

			std::vector<double> inputs;
			int iter = 0;

			for (csv::CSVField& field : row)
			{
				double v = field.get<double>();
				iter++;

				if (iter == 1)
				{
					targets[v] = 1.0;
					continue;
				}
				else
				{
					// v = (v > 128) ? 1.0 : 0.0;
					v /= 255.0;
				}

				inputs.push_back(v);
			}

			double sum = 0.0;

			neuralNet->Train(inputs, targets);
		}

		trainData.empty();
		std::cout << "Epoch " << e << " / Training is done." << std::endl;

		double ssr = 0.0;
		double tss = 0.0;

		csv::CSVReader testData("datasets/input/mnist_test.csv");

		for (csv::CSVRow& row : testData)
		{
			std::vector<double> targets;
			for (int i = 0; i < 10; i++) targets.push_back(0.0);

			std::vector<double> inputs;
			int iter = 0;

			int tar = 0;

			for (csv::CSVField& field : row)
			{
				double v = field.get<double>();
				iter++;

				if (iter == 1)
				{
					targets[v] = 1.0;
					tar = v;
					continue;
				}
				else
				{
					// v = (v > 128) ? 1.0 : 0.0;
					v /= 255.0;
				}

				inputs.push_back(v);
			}

			std::vector<double> out = neuralNet->Forward(inputs);

			double highestVal = 0.0;
			int highestT = 0;

			for (int i = 0; i < 10; i++)
			{
				double e = (targets[i] - out[i]);
				double m = (targets[i] - 0.1);
				ssr += e * e;
				tss += m * m;

				if (out[i] > highestVal)
				{
					highestVal = out[i];
					highestT = i;
				}
			}

			// if (e > 0) std::cout << "Epoch " << e << " / Prediction: " << highestT << " / Actual: " << tar << std::endl;
		}

		testData.empty();
		std::cout << "Epoch " << e << " / Accuracy: " << (1.0 - (ssr / tss)) << std::endl;
	}

	delete neuralNet;
}