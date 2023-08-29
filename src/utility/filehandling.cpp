#include "filehandling.h"

std::wstring ReadFile(std::string path)
{
	std::wifstream file;
	std::wstring data;
	try
	{
		file.open(path);
		if (file.fail()) throw 1;

		std::wstringstream buff;
		buff << file.rdbuf();
		data = buff.str();

		file.close();
	}
	catch (int id)
	{
		std::wcout << L"The file could not be found at the specified path.\n"
			<< L"If you haven't already, move the datasets folder into the build folder and try again.\n"
			<< L"Otherwise, throw up your hands (for now, at least).\n\n";
	}

	return data;
}

void WriteFile(std::string path, std::wstring data)
{
	std::wofstream file(path);
	file.clear();
	file << data;
	file.close();
}