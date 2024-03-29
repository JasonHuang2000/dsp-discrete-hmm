#include <hmm.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

typedef vector<int> vi;
typedef vector<vector<int>> vvi;
typedef vector<double> vd;
typedef vector<vector<double>> vvd;
typedef vector<vector<vector<double>>> vvvd;
typedef vector<string> vs;

struct Variable {
    Variable(HMM* h, int seq_len) {
        hmm = h;
        alpha = vvd(seq_len, vd(h->state_num));
        beta = vvd(seq_len, vd(h->state_num));
        gamma = vvd(seq_len, vd(h->state_num));
        epsilon = vvvd(seq_len - 1, vvd(h->state_num, vd(h->state_num)));
    }
    void calc_alpha(vi& seq) {
        int seq_len = seq.size();
        // initialization
        for (int i = 0; i < hmm->state_num; ++i) {
            alpha[0][i] = hmm->initial[i] * hmm->observation[seq[0]][i];
        }
        // induction
        for (int i = 1; i < seq_len; ++i) {
            for (int j = 0; j < hmm->state_num; ++j) {
                alpha[i][j] = 0.0;
                for (int k = 0; k < hmm->state_num; ++k) {
                    // i := current sequence index
                    // j := current state
                    // k := previous state
                    alpha[i][j] += (alpha[i - 1][k] * hmm->transition[k][j]);
                }
                alpha[i][j] *= hmm->observation[seq[i]][j];
            }
        }
    }
    void calc_beta(vi& seq) {
        int seq_len = seq.size();
        // initialization
        for (int i = 0; i < hmm->state_num; ++i) {
            beta[seq_len - 1][i] = 1;
        }
        // induction
        for (int i = seq_len - 2; i >= 0; --i) {
            for (int j = 0; j < hmm->state_num; ++j) {
                beta[i][j] = 0.0;
                for (int k = 0; k < hmm->state_num; ++k) {
                    // i := current sequence index
                    // j := current state
                    // k := next state
                    beta[i][j] += (beta[i + 1][k] * hmm->transition[j][k] *
                                   hmm->observation[seq[i + 1]][k]);
                }
            }
        }
    }
    void calc_gamma(vi& seq) {
        int seq_len = seq.size();
        for (int i = 0; i < seq_len; ++i) {
            double sum = .0;
            for (int j = 0; j < hmm->state_num; ++j) {
                // i := current sequence index
                // j := current state
                sum += (alpha[i][j] * beta[i][j]);
            }
            for (int j = 0; j < hmm->state_num; ++j) {
                gamma[i][j] = (alpha[i][j] * beta[i][j]) / sum;
            }
        }
    }
    void calc_epsilon(vi& seq) {
        int seq_len = seq.size();
        for (int i = 0; i < seq_len - 1; ++i) {
            double sum = 0.0;
            for (int j = 0; j < hmm->state_num; ++j) {
                for (int k = 0; k < hmm->state_num; ++k) {
                    // i := current sequence index
                    // j := current state
                    // k := next state
                    sum += (alpha[i][j] * hmm->transition[j][k] *
                            hmm->observation[seq[i + 1]][k] * beta[i + 1][k]);
                }
            }
            for (int j = 0; j < hmm->state_num; ++j) {
                for (int k = 0; k < hmm->state_num; ++k) {
                    epsilon[i][j][k] =
                        (alpha[i][j] * hmm->transition[j][k] *
                         hmm->observation[seq[i + 1]][k] * beta[i + 1][k]) /
                        sum;
                }
            }
        }
    }
    void calc_vars(vi& seq) {
        // calculate multiple variables
        calc_alpha(seq);
        calc_beta(seq);
        calc_gamma(seq);
        calc_epsilon(seq);
    }
    HMM* hmm;
    vvd alpha;
    vvd beta;
    vvd gamma;
    vvvd epsilon;
};

struct Sum {
    Sum(HMM* h, Variable* v) {
        hmm = h;
        vars = v;
        init_gamma_sum = vd(hmm->state_num);
        gamma_sum = vd(hmm->state_num);
        gamma_sum_for_b = vd(hmm->state_num);
        gamma_observation_sum = vvd(hmm->observ_num, vd(hmm->state_num));
        epsilon_sum = vvd(hmm->state_num, vd(hmm->state_num));
    }
    void init() {
        fill(init_gamma_sum.begin(), init_gamma_sum.end(), .0);
        fill(gamma_sum.begin(), gamma_sum.end(), .0);
        fill(gamma_sum_for_b.begin(), gamma_sum_for_b.end(), .0);
        for (vd& v : gamma_observation_sum) {
            fill(v.begin(), v.end(), .0);
        }
        for (vd& v : epsilon_sum) {
            fill(v.begin(), v.end(), .0);
        }
    }
    void update_sum(vi& seq) {
        int seq_len = seq.size();
        for (int i = 0; i < hmm->state_num; ++i) {
            init_gamma_sum[i] += vars->gamma[0][i];
            for (int t = 0; t < seq_len - 1; ++t) {
                gamma_sum[i] += vars->gamma[t][i];
                for (int j = 0; j < hmm->state_num; ++j) {
                    epsilon_sum[i][j] += vars->epsilon[t][i][j];
                }
            }
        }
        for (int i = 0; i < hmm->state_num; ++i) {
            for (int t = 0; t < seq_len; ++t) {
                gamma_sum_for_b[i] += vars->gamma[t][i];
                gamma_observation_sum[seq[t]][i] += vars->gamma[t][i];
            }
        }
    }
    void update_model_param(int N) {
        for (int i = 0; i < hmm->state_num; ++i) {
            hmm->initial[i] = init_gamma_sum[i] / N;
            for (int j = 0; j < hmm->state_num; ++j) {
                hmm->transition[i][j] = epsilon_sum[i][j] / gamma_sum[i];
            }
            for (int k = 0; k < hmm->observ_num; ++k) {
                hmm->observation[k][i] =
                    gamma_observation_sum[k][i] / gamma_sum_for_b[i];
            }
        }
    }
    HMM* hmm;
    Variable* vars;
    vd init_gamma_sum;
    vd gamma_sum;
    vd gamma_sum_for_b;
    vvd gamma_observation_sum;
    vvd epsilon_sum;
};

void train(vvi& seq_arr, HMM& hmm, int iter) {
    // length of sequence
    int seq_len = seq_arr[0].size();

    // initialize variables
    Variable vars(&hmm, seq_len);
    Sum sum(&hmm, &vars);

    for (int step = 0; step < iter; ++step) {
        sum.init();
        for (vi& seq : seq_arr) {
            vars.calc_vars(seq);
            sum.update_sum(seq);
        }
        // update model
        sum.update_model_param(seq_arr.size());
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: ./train <iter> <model_init_path> <seq_path> "
                "<output_model_path>\n";
        exit(1);
    }

    // command line parameters
    int iter = atoi(argv[1]);
    char* model_init_path = argv[2];
    char* seq_path = argv[3];
    char* output_model_path = argv[4];

    // load initial model
    HMM hmm;
    loadHMM(&hmm, model_init_path);

    // traning data
    ifstream training_data(seq_path, ifstream::in);
    vvi seq_arr;
    string seq_str;
    while (training_data >> seq_str) {
        vi seq;
        for (char c : seq_str) {
            // transform to integer
            seq.push_back(c - 'A');
        }
        seq_arr.push_back(seq);
    }
    training_data.close();

    // traning process
    if (!seq_arr.empty()) {
        train(seq_arr, hmm, iter);
    }

    // dump model
    FILE* fp = fopen(output_model_path, "w");
    dumpHMM(fp, &hmm);
    fclose(fp);

    return 0;
}