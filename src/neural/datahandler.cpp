#include "datahandler.h"
#include "../utility/filehandling.h"
#include <queue>


void DataHandler::ReadEmbeddings(std::string path)
{
	std::wstring input = WReadFile(path);

	unsigned int embeddingCount = 0;

	bool syl = true;
	std::wstring syllableText = L"";
	std::wstring text = L" ";
	unsigned int count = 0;

	double v[network->EmbeddingSize]{0};

	for (int i = 0; i < input.size(); i++)
	{
		wchar_t c = input[i];

		if (c == L' ')
		{
			if (syl)
			{
				syl = false;

				syllableText = text;

				text = L"";

				continue;
			}

			v[count] = wcstod(text.c_str(), NULL);

			text = L"";

			count++;
		}
		else if (c == L'\n' || i == input.size() - 1)
		{	
			syl = true;

			v[count] = wcstod(text.c_str(), NULL);

			count = 0;

			Embedding emb =
			{
				indexMap[syllableText],
				syllableText
			};

			embeddingCount++;

			for (int j = 0; j < network->EmbeddingSize; j++) emb.vector[j] = v[j];

			embMap[syllableText] = emb;
			
			syllableText = L"";
			text = L"";

			continue;
		}

		text += c;
	}
}

std::wstring DataHandler::NextTargetSyllable()
{
	std::wstring syllable;
	unsigned int start = targetCursor;

	while (true)
	{
		wchar_t c = syllabizedTokenizedCorpus[targetCursor];
		wchar_t cTwo = syllabizedTokenizedCorpus[targetCursor + 1];
		targetCursor++;

		if (c == L'\n') continue;

		if (cTwo == L'\n')
		{
			unsigned int index = indexMap[syllable];

			// Hypothetically this could be a problem, but
			// I'll leave it as is for now.
			/*if (index == 0 && syllable != sylMap[0])
			{
				syllable = L"";
				continue;
			}*/

			syllable += c;
			return syllable;
		}

		syllable += c;
	}
}

std::wstring DataHandler::NextInputSyllable()
{
	std::wstring syllable;
	unsigned int start = inputCursor;

	while (true)
	{
		wchar_t c = syllabizedTokenizedCorpus[inputCursor];
		wchar_t cTwo = syllabizedTokenizedCorpus[inputCursor + 1];
		inputCursor++;

		if (c == L'\n') continue;

		if (cTwo == L'\n')
		{
			unsigned int index = indexMap[syllable];

			/*if (index == 0 && syllable != sylMap[0])
			{
				syllable = L"";
				continue;
			}*/

			return syllable;
		}

		syllable += c;
	}
}

void DataHandler::Syllabize()
{
	std::wstring syllable;
	std::wstring syllabized;

	for (int i = 0; i < tokenizedCorpus.size(); i++)
	{
		wchar_t c = tokenizedCorpus[i];

		if (c == L'་' || c == L' ')
		{
			if (freqMap.find(syllable) == freqMap.end())
			{
				freqMap.emplace(syllable, 1);
				tempMap[nUniqueSyllables] = syllable;
				nUniqueSyllables++;
			}
			else
			{
				freqMap[syllable]++;
			}

			nTotalSyllables++;
			syllabized += syllable + c + L"\n";
			syllable = L"";
			continue;
		}

		syllable += c;
	}

	/*
		Now we go through and remove the syllables with
		a frequency less than some number.
	*/
	std::wstring uniqsyls = L"";
	unsigned int oldCount = nUniqueSyllables;
	unsigned int count = 0;

	unsigned int mostFrequentValue = 0;
	unsigned int mostFrequentIndex = 0;

	for (int i = 0; i < oldCount; i++)
	{
		syllable = tempMap[i];
		unsigned int freq = freqMap[syllable];

		if (freq < 5 || syllable == L"")
		{
			if (freq > mostFrequentValue)
			{
				mostFrequentValue = freq;
				mostFrequentIndex = i;
			}

			freqMap.erase(syllable);
			nUniqueSyllables--;
		}
		else
		{
			sylMap.emplace(count, syllable);
			indexMap.emplace(syllable, count);
			count++;
			uniqsyls += syllable + L"\n";
		}
	}

	WriteFile("datasets/output/unique_syllables.txt", uniqsyls);
	WriteFile("datasets/output/syllabized_corpus.txt", syllabized);
	std::cout << "Unique syllables: " << nUniqueSyllables << "\n";
	std::cout << "Total syllables: " << nTotalSyllables << "\n\n";

	syllabizedTokenizedCorpus = syllabized;
}

void DataHandler::Train()
{
	unsigned int maxEpochs = 20;

	for (int e = 0; e < maxEpochs; e++)
	{
		std::cout << "Beginning training for epoch #" << e << "\n";

		// Let's throw some variables here to keep track of stuff.
		unsigned int syllablesCovered = 0;
		unsigned int inputWings = network->WindowSize / 2;
		double mean = 0.0;
		unsigned int meanCount = 0;

		/*
			First, let's find our first set of inputs.
		*/
		std::vector<double> inputs(network->InputCount);

		std::vector<unsigned int> inputIndices(network->WindowSize + 1);

		unsigned int inputIndexCursor = 0;
		for (int i = 0; i < network->WindowSize + 1; i++)
		{
			std::wstring syl = NextInputSyllable();
			unsigned int index = indexMap[syl];
			syllablesCovered++;

			if (i == inputWings)
			{
				targetCursor = inputCursor;
				// continue;
			}

			Embedding* emb = &embMap[syl];
			for (int j = 0; j < network->EmbeddingSize; j++) inputs[inputIndexCursor++] = emb->vector[j];

			inputIndices[i] = index;
		}

		/*
			Now we'll grab our first target.
		*/
		std::wstring targSyl = NextTargetSyllable();
		double target = (targSyl[targSyl.size() - 1] != L' ');

		/*
			Now, we slide over the rest of our corpus.
		*/

		unsigned int maxTrainCount = nTotalSyllables * 0.8;
		unsigned int maxTestCount = nTotalSyllables * 0.2;

		std::wstring test = L"";

		while (syllablesCovered < maxTrainCount)
		{
			// Let's run our training.
			network->Train(inputs, { target });

			// Now, we need to load in our next input.
			std::wstring inSyl = NextInputSyllable();
			unsigned int index = indexMap[inSyl];
			syllablesCovered++;

			for (int i = 0; i < network->WindowSize; i++) inputIndices[i] = inputIndices[i + 1];
			inputIndices[network->WindowSize] = index;

			std::wstring context = L"";
			inputIndexCursor = 0;
			for (int i = 0; i < inputIndices.size(); i++)
			{	
				std::wstring iisyl = sylMap[inputIndices[i]];
				Embedding embed = embMap[iisyl];

				context += iisyl + L" : ";

				for (int j = 0; j < network->EmbeddingSize; j++) inputs[inputIndexCursor++] = embed.vector[j];
			}

			// And now we need to grab our next target.
			targSyl = NextTargetSyllable();
			target = (targSyl[targSyl.size() - 1] != L' ');
			mean += target;
			meanCount++;

			test += targSyl + L" : " + std::to_wstring(target) + L" // " + context + L"\n";

			// Let's print for test.
			if (syllablesCovered % 1000 == 0)
			{
				std::cout << "\r" << syllablesCovered << " / " << maxTrainCount;
			}
		}

		WriteFile("datasets/output/train-" + std::to_string(e) + ".txt", test);

		std::cout << "\nBeginning testing for epoch #" << e << "\n";

		mean /= meanCount;
		double ssr = 0.0;
		double tss = 0.0;

		std::wstring predTest;

		unsigned int totalCorrect = 0;
		unsigned int totalChecked = 0;

		while (syllablesCovered <= maxTrainCount + maxTestCount && syllablesCovered < nTotalSyllables)
		{
			// Let's run our test.
			std::vector<double> predictions = network->Forward(inputs);

			std::wstring targVal = std::to_wstring(target);
			std::wstring predVal = std::to_wstring(predictions[0]);

			// Now we'll calc the diff.
			double p = (predictions[0] >= 0.5) ? 1.0 : 0.0;

			if (target == p) totalCorrect++;
			totalChecked++;

			double e = target - p;
			double m = target - mean;
			ssr += e * e;
			tss += m * m;

			// Now, we need to load in our next input.
			std::wstring inSyl = NextInputSyllable();
			double index = indexMap[inSyl];
			syllablesCovered++;

			for (int i = 0; i < network->WindowSize; i++) inputIndices[i] = inputIndices[i + 1];
			inputIndices[network->WindowSize] = index;
			
			std::wstring context = L"";
			inputIndexCursor = 0;
			for (int i = 0; i < inputIndices.size(); i++)
			{	
				std::wstring iisyl = sylMap[inputIndices[i]];
				Embedding embed = embMap[iisyl];

				context += iisyl + L" : ";

				for (int j = 0; j < network->EmbeddingSize; j++) inputs[inputIndexCursor++] = embed.vector[j];
			}

			predTest += targSyl + L" : " + targVal + L" // " + predVal + L" // " + context + L"\n";

			// And now we need to grab our next target.
			targSyl = NextTargetSyllable();
			target = (targSyl[targSyl.size() - 1] != L' ');

			// Let's print for test.
			if (syllablesCovered % 1000 == 0)
			{
				std::cout << "\r" << syllablesCovered << " / " << maxTrainCount + maxTestCount;
			}
		}

		double estAcc = (totalCorrect / (double)totalChecked);
		WriteFile("datasets/output/test-" + std::to_string(e) + ".txt", predTest);
		std::cout << "\nFinished epoch #" << e << " / Accuracy: " << estAcc << "\n\n";
		syllablesCovered = 0;
		targetCursor = 0;
		inputCursor = 0;
	}
}

DataHandler::DataHandler(std::wstring nontoken, std::wstring token, std::vector<LayerType> hiddenLayerTypes, std::vector<PoolType> poolTypes)
{
	this->nUniqueSyllables = 0;
	this->nTotalSyllables = 0;

	this->targetCursor = 0;
	this->inputCursor = 0;

	this->nontokenizedCorpus = nontoken;
	this->tokenizedCorpus = token;
	Syllabize();

	ReadEmbeddings("datasets/input/bod_vectors.txt");

	this->network = new NeuralNet(hiddenLayerTypes, poolTypes);
}


DataHandler::~DataHandler()
{
	delete network;
}