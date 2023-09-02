#include "scrubber.h"

std::wstring Scrub(std::wstring in)
{
	std::wstring out;
	wchar_t lastChar = L' ';

	for (int i = 0; i < in.size(); i++)
	{
		int val = (int)in[i];

		// We'll see if this actually works.
		if (val == 32 || (val >= (int)L'༠' && val < (int)L'྾'))
		{
			if (val == 32 && lastChar == L' ') continue;

			out += in[i];
			lastChar = in[i];
		}
	}

	return out;
}

std::wstring Syllabize(std::wstring in)
{
	return L"";
}