/*
	main.cpp
*/

#include <chrono>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>

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
		<<"You are running the Lhasa Tibetan NNTokenizer.\n"
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
	std::wstring nontoken = Scrub(ReadFile(NONTOKENPATH));
	std::wstring token = Scrub(ReadFile(TOKENPATH));

	std::wcout << "The nontokenized corpus contains " << nontoken.size() << " characters.\n";
	std::wcout << "And the tokenized corpus contains " << token.size() << " characters.\n\n";

	WriteFile("datasets/output/scrubbed_nontokenized.txt", nontoken);
	WriteFile("datasets/output/scrubbed_tokenized.txt", token);

	/*
		Now, we need to go through and start preparing both
		corpora, transferring them into data structures intelligible
		to the neural network.
	*/

	NeuralNet* neuralNet = new NeuralNet();
	DataHandler* dataHandler = new DataHandler(neuralNet, nontoken, token);

	dataHandler->Train();

	delete neuralNet;
	delete dataHandler;

	/*NeuralNet* neuralNet = new NeuralNet();

	std::vector<double> trainingInputs;
	std::vector<double> trainingTargets;

	for (int i = 0; i < NeuralNet::InputCount; i++)
	{
		trainingInputs.push_back(WCharToDouble(token[i]));
		std::cout << std::to_string(WCharToDouble(token[i])) << std::endl;
	}
	for (int i = 0; i < NeuralNet::OutputCount; i++)
	{
		trainingTargets.push_back(WCharToDouble(token[i]));
		std::cout << std::to_string(WCharToDouble(token[i])) << std::endl;
	}

	int trainingIters = 1000;

	for (int i = 0; i < trainingIters; i++)
	{
		neuralNet->Train(trainingInputs, trainingTargets, false);
	}

	double mean = 0.0;
	for (int i = 0; i < trainingTargets.size(); i++) mean += trainingTargets[i];
	mean /= trainingTargets.size();

	double ssr = 0.0;
	double tss = 0.0;

	int count = 0;
	for (int i = 0; i < trainingIters; i ++)
	{
		std::vector<double> out = neuralNet->Forward(trainingInputs, false);

		for (int j = 0; j < out.size(); j++)
		{
			ssr += (trainingTargets[j] - out[j]) * (trainingTargets[j] - out[j]);
			tss += (trainingTargets[j] - mean) * (trainingTargets[j] - mean);
		}
	}

	std::cout << "R^2: " << (1.0 - (ssr / tss)) << std::endl;

	delete neuralNet;*/
}