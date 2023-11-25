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
	std::wstring nontoken = Scrub(WReadFile(NONTOKENPATH));
	std::wstring token = Scrub(WReadFile(TOKENPATH));

	std::wcout << "The nontokenized corpus contains " << nontoken.size() << " characters.\n";
	std::wcout << "And the tokenized corpus contains " << token.size() << " characters.\n\n";

	WriteFile("datasets/output/scrubbed_nontokenized.txt", nontoken);
	WriteFile("datasets/output/scrubbed_tokenized.txt", token);

	/*
		Now, we need to go through and start preparing both
		corpora, transferring them into data structures intelligible
		to the neural network.
	*/

	DataHandler* dataHandler = new DataHandler(nontoken, token, { LayerType::hidden, LayerType::hidden, LayerType::hidden, LayerType::hidden, LayerType::hidden, LayerType::hidden }, { });

	dataHandler->Train();

	delete dataHandler;
}