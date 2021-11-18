#include <hmm.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

#define MAX_MODEL_NUM 5

typedef vector<double> vd;
typedef vector<vector<double>> vvd;
typedef vector<int> vi;
typedef vector<vector<int>> vvi;

struct Variable {
    Variable(HMM* h, int seq_len) {
        hmm = h;
        delta = vvd(seq_len, vd(h->state_num));
    }
    double calc_data_return_max(vi& seq) {
        int seq_len = seq.size();
        // initialize
        for (int i = 0; i < hmm->state_num; ++i) {
            delta[0][i] = hmm->initial[i] * hmm->observation[seq[0]][i];
        }
        // induction
        for (int t = 1; t < seq_len; ++t) {
            for (int j = 0; j < hmm->state_num; ++j) {
                double max_prob = .0;
                for (int i = 0; i < hmm->state_num; ++i) {
                    // t := current seqence index
                    // j := current state
                    // i := previous state
                    max_prob =
                        max(max_prob, delta[t - 1][i] * hmm->transition[i][j]);
                }
                delta[t][j] = max_prob * hmm->observation[seq[t]][j];
            }
        }
        // find the maximum of the last column
        double max_path_prob = .0;
        for (int i = 0; i < hmm->state_num; ++i) {
            max_path_prob = max(max_path_prob, delta[seq_len - 1][i]);
        }
        return max_path_prob;
    }
    HMM* hmm;
    vvd delta;
};

struct Result {
    Result(int idx, double p) {
        model_idx = idx;
        prob = p;
    }
    int model_idx;
    double prob;
};

void test(HMM* hmms, vvi& seq_arr, vector<Result>& res) {
    // length of the sequence
    int seq_len = seq_arr[0].size();

    // initialize variables for each model
    vector<Variable> vars;
    for (int i = 0; i < MAX_MODEL_NUM; ++i) {
        vars.push_back(Variable(&(hmms[i]), seq_len));
    }

    // loop through all the sequence
    for (vi& seq : seq_arr) {
        double max_model_prob = .0, cur_model_prob;
        int max_model_idx;
        for (int i = 0; i < MAX_MODEL_NUM; ++i) {
            cur_model_prob = vars[i].calc_data_return_max(seq);
            if (cur_model_prob > max_model_prob) {
                max_model_prob = cur_model_prob;
                max_model_idx = i;
            }
        }
        res.push_back(Result(max_model_idx, max_model_prob));
    }
}

int main(int argc, char* argv[]) {
    // check parameters
    if (argc != 4) {
        cerr << "Usage: ./test <models_list_path> <seq_path> "
                "<output_result_path>\n";
        exit(1);
    }

    // process parameters
    char* model_list_path = argv[1];
    char* seq_path = argv[2];
    char* output_result_path = argv[3];

    // load models
    HMM hmms[MAX_MODEL_NUM];
    load_models(model_list_path, hmms, MAX_MODEL_NUM);

    // read in seqence
    ifstream testing_data(seq_path, fstream::in);
    vvi seq_arr;
    string seq_str;
    while (testing_data >> seq_str) {
        vi seq;
        for (char c : seq_str) {
            seq.push_back(c - 'A');
        }
        seq_arr.push_back(seq);
    }

    // testing process
    vector<Result> res;
    test(hmms, seq_arr, res);

    // dump result to target file
    ofstream prediction(output_result_path, fstream::out);
    for (Result& r : res) {
        prediction << "model_0" << r.model_idx + 1 << ".txt " << r.prob << endl;
    }
    prediction.close();

    return 0;
}