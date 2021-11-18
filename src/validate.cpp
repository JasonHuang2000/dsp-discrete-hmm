#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: ./validate <result_file> <label_file>\n";
        exit(1);
    }
    ifstream result_file(argv[1], fstream::in);
    ifstream label_file(argv[2], fstream::in);
    string s1, s2;
    double p;
    int sum = 0, cnt = 0;
    while (result_file >> s1 >> p) {
        label_file >> s2;
        if (s1 == s2) {
            cnt++;
        }
        sum++;
    }
    cout << "Accuracy: " << (double)cnt / sum * 100 << "%\n";
    return 0;
}