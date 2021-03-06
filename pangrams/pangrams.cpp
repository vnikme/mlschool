#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <locale>
#include <string>
#include <string_view>
#include <ctime>



const std::string ALPHABET = "abcdefghijklmnopqrstuvwxyz";


void ToLower(std::string &text) {
    std::locale locale;
    std::transform(text.begin(), text.end(), text.begin(), [&locale](char sym) { return std::tolower(sym, locale); });
}


void RunAndMeasure(const std::function<bool(const std::string &)> &func, int count, std::string text) {
    ToLower(text);
    std::cout << (func(text) ? "true" : "false") << ' ';
    std::clock_t start = std::clock();
    for (int i = 0; i < count; ++i)
        func(text);
    std::clock_t finish = std::clock();
    std::cout << (finish - start) << std::endl;
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


bool CheckForEveryDigitWithoutLower(const std::string_view &text) {
    for (char sym : ALPHABET) {
        if (text.find(sym) == std::string::npos)
            return false;
    }
    return true;
}


bool CheckViaFlags(const std::string_view &text) {
    int flags = 0, count = 0;
    for (char sym : text) {
        unsigned char code = sym - 'a';
        if (code < 26) {
            int mask = (1 << code);
            if ((flags & mask) == 0) {
                flags |= (1 << code);
                if (++count == 26)
                    return true;
            }
        }
    }
    return false;
}


void RunAll(int count, const std::string &text) {
    std::cout << "Text size: " << text.size() << std::endl;
    //RunAndMeasure(CheckForEveryDigit, count, text);
    RunAndMeasure(CheckForEveryDigitWithoutLower, count, text);
    RunAndMeasure(CheckViaFlags, count, text);
    std::cout << std::endl;
}


void TestLongString(int paddingSize) {
    std::string text;
    for (int i = 0; i < paddingSize; ++i) {
        text += ' ';
        if (rand() % 2 == 0)
            text += '.';
        else
            text += ',';
    }
    for (int i = 0; i < ALPHABET.size(); ++i) {
        for (int j = 10; j > 0; --j) {
            text += ALPHABET[i];
        }
    }
    RunAll(1, text);

}


int main() {
    RunAll(10000, "Jackdaws love my big sphinx of quartz");
    //RunAll(10000, "Jackdaws love my sphinx of quartz");
    TestLongString(1000000);
    TestLongString(10000000);
    return 0;
}

