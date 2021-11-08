#include <hmm.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

typedef vector<vector<double>> vvb;
typedef vector<vector<vector<double>>> vvvb;

void calc_alpha(HMM& hmm, vvb& alpha, string& seq) {
    int seq_len = seq.size();
    // initialization
    for (int i = 0; i < hmm.state_num; ++i) {
        alpha[0][i] = hmm.initial[i] * hmm.observation[seq[0]][i];
    }
    // induction
    for (int i = 1; i < seq_len; ++i) {
        for (int j = 0; j < hmm.state_num; ++j) {
            alpha[i][j] = 0.0;
            for (int k = 0; k < hmm.state_num; ++k) {
                // i := current sequence index
                // j := current state
                // k := previous state
                alpha[i][j] += (alpha[i - 1][k] * hmm.transition[k][j]);
            }
            alpha[i][j] *= hmm.observation[seq[i]][j];
        }
    }
}

void calc_beta(HMM& hmm, vvb& beta, string& seq) {
    int seq_len = seq.size();
    // initialization
    for (int i = 0; i < hmm.state_num; ++i) {
        beta[seq_len - 1][i] = 1;
    }
    // induction
    for (int i = seq_len - 2; i >= 0; --i) {
        for (int j = 0; j < hmm.state_num; ++j) {
            beta[i][j] = 0.0;
            for (int k = 0; k < hmm.state_num; ++k) {
                // i := current sequence index
                // j := current state
                // k := next state
                beta[i][j] += (beta[i + 1][k] * hmm.transition[j][k] *
                               hmm.observation[seq[i]][j]);
            }
        }
    }
}

void calc_gamma(HMM& hmm, vvb& alpha, vvb& beta, vvb& gamma, int seq_len) {
    for (int i = 0; i < seq_len; ++i) {
        double sum = 0.0;
        for (int j = 0; j < hmm.state_num; ++j) {
            // i := current sequence index
            // j := current state
            sum += (alpha[i][j] * beta[i][j]);
        }
        for (int j = 0; j < hmm.state_num; ++j) {
            gamma[i][j] = (alpha[i][j] * beta[i][j]) / sum;
        }
    }
}

void calc_epsilon(HMM& hmm, vvb& alpha, vvb& beta, vvvb& epsilon, string& seq) {
    int seq_len = seq.size();
    for (int i = 0; i < seq_len - 1; ++i) {
        double sum = 0.0;
        for (int j = 0; j < hmm.state_num; ++j) {
            for (int k = 0; k < hmm.state_num; ++k) {
                // i := current sequence index
                // j := current state
                // k := next state
                sum += (alpha[i][j] * hmm.transition[j][k] *
                        hmm.observation[seq[i + 1]][k] * beta[i + 1][k]);
            }
        }
        for (int j = 0; j < hmm.state_num; ++j) {
            for (int k = 0; k < hmm.state_num; ++k) {
                epsilon[i][j][k] =
                    (alpha[i][j] * hmm.transition[j][k] *
                     hmm.observation[seq[i + 1]][k] * beta[i + 1][k]) /
                    sum;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: ./train <iter> <model_init_path> <seq_path> "
                "<output_model_path>\n";
        exit(1);
    }
    HMM hmm_initial;
    loadHMM(&hmm_initial, "../model_init.txt");
    return 0;
}