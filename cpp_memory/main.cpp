#include <ctime>
#include <iostream>
#include <vector>


void IntParamsVal(int a, int b) {
    std::cout << a / b << std::endl;
    a = 1;
    b = 2;
}


void TestIntVal() {
    IntParamsVal(6, 3);
    int a = 3, b = 6;
    IntParamsVal(b, a);
    std::cout << a << ' ' << b << std::endl;
}


void IntParamsRef(int &a, int &b) {
    std::cout << a / b << std::endl;
    a = 1;
    b = 2;
}


void TestIntRef() {
    int a = 3, b = 6;
    IntParamsRef(b, a);
    std::cout << a << ' ' << b << std::endl;
}


void TestOneArray(int a[]) {
    a[3] = 777;
}


void TestArray() {
    int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    TestOneArray(a);
    //int b = 678;
    //TestOneArray(&b);
    for (int i = 0; i < 10; ++i)
        std::cout << a[i] << ' ';
    std::cout << std::endl;
}


void TestOneVectorVal(std::vector<int> data) {
    std::cout << data.size() << std::endl;
}


void TestVectorVal() {
    time_t start = time(nullptr);
    std::vector<int> data(500000000, -1);
    for (int i = 0; i < 10; ++i)
        TestOneVectorVal(data);
    std::cout << (time(nullptr) - start) << std::endl;
}


void TestOneVectorRef(const std::vector<int> &data) {
    std::cout << data.size() << std::endl;
}


void TestVectorRef() {
    time_t start = time(nullptr);
    std::vector<int> data(500000000, -1);
    for (int i = 0; i < 10; ++i)
        TestOneVectorRef(data);
    std::cout << (time(nullptr) - start) << std::endl;
}


std::vector<int> TestRVO() {
    std::vector<int> result;
    for (int i = 0; i < 10000000; +i)
        result.push_back(i);
    return result;
}


int main() {
    TestIntVal();
    TestIntRef();
    TestArray();
    //TestVectorVal();
    //TestVectorRef();
    auto data = TestRVO();
    return 0;
}

