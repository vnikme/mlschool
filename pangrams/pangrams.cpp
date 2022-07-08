#include <algorithm>
#include <iostream>
#include <locale>
#include <string>
#include <ctime>



const std::string ALPHABET = "abcdefghijklmnopqrstuvwxyz";


template<typename TFunc>
void RunAndMeasure(const TFunc &func, int count, const std::string &text) {
    std::cout << (func(text) ? "true" : "false") << ' ';
    std::clock_t start = std::clock();
    for (int i = 0; i < count; ++i)
        func(text);
    std::clock_t finish = std::clock();
    std::cout << (finish - start) << std::endl;
}


void ToLower(std::string &text) {
    std::locale locale;
    std::transform(text.begin(), text.end(), text.begin(), [&locale](char sym) { return std::tolower(sym, locale); });
}


bool CheckForEveryDigit(const std::string &text) {
    for (char sym : ALPHABET) {
        std::string lowText(text);
        ToLower(lowText);
        if (lowText.find(sym) == std::string::npos)
            return false;
    }
    return true;
}


bool CheckForEveryDigitWithoutLower(const std::string &text) {
    //ToLower(text);
    for (char sym : ALPHABET) {
        if (text.find(sym) == std::string::npos)
            return false;
    }
    return true;
}


bool CheckViaFlags(const std::string &text) {
    //ToLower(text);
    int flags = 0;
    for (char sym : text) {
        unsigned char code = int(sym) - int('a');
        if (code < 26)
            flags |= (1 << code);
    }
    return (flags == (1 << 26) - 1);
}


bool CheckViaFlagsImproved(const std::string &text) {
    //ToLower(text);
    int trueCount = 0;
    bool flags[26] = {};
    for (char sym : text) {
        int code = int(sym) - int('a');
        if (0 <= code && code < 26) {
            if (!flags[code]) {
                flags[code] = true;
                ++trueCount;
            }
        }
    }
    return (trueCount == 26);
}



void RunAll(int count, const std::string &text) {
    std::cout << "Text size: " << text.size() << std::endl;
    RunAndMeasure(CheckForEveryDigit, count, text);
    RunAndMeasure(CheckForEveryDigitWithoutLower, count, text);
    RunAndMeasure(CheckViaFlags, count, text);
    RunAndMeasure(CheckViaFlagsImproved, count, text);
    std::cout << std::endl;
}


int main() {
    RunAll(10000, "Jackdaws love my big sphinx of quartz");
    RunAll(10000, "Jackdaws love my sphinx of quartz");
    std::string text;
    for (int i = 0; i < 1000000; ++i)
        text += ' ';
    for (int i = 0; i < ALPHABET.size(); ++i) {
        for (int j = 10; j > 0; --j) {
            text += ALPHABET[i];
        }
    }
    RunAll(1, text);
    return 0;
}

