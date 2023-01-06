#include <ctime>
#include <iostream>
#include <list>
#include <vector>


template<typename T>
void Test(int n) {
    T container;
    for (int i = 0; i < n; ++i)
        container.push_back(i);
    while (!container.empty())
        container.erase(container.begin());
        //container.pop_back();
}


template<typename T>
void TestAndMeasure(int n, const std::string &name) {
    time_t start = time(nullptr);
    Test<T>(n);
    std::cout << name << " tested: " << time(nullptr) - start << std::endl;
}


int main() {
    for (int n = 10; n <= 1000000; n *= 2) {
        std::cout << "Size: " << n << std::endl;
        TestAndMeasure<std::list<int>>(n, "List");
        TestAndMeasure<std::vector<int>>(n, "Vector");
    }
    return 0;
}

