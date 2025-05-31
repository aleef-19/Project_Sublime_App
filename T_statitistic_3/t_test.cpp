#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

using namespace std;

Eigen::VectorXd T_static(
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &beta) {

        int num_samples = X.rows();
        int num_features = X.cols();

        Eigen::VectorXd residuals =  y - X * beta;

        double mse = residuals.squaredNorm() / (num_samples - num_features);

        Eigen::MatrixXd XtX_inv = (X.transpose() * X).inverse();

        Eigen::VectorXd t_stats(num_features);

        for(int i = 0; i < num_features; ++i){
            double std_error = sqrt(mse * XtX_inv(i, i));
            t_stats(i) = beta(i) / std_error;
        } 

        return t_stats;
}

int main(){
    ifstream input_file("data.dat");
    if(!input_file){
        cerr << "Cannot open file\n";
        return 1;
    }

    string line;

    getline(input_file, line);

    int num_line = 0;
    streampos pos = input_file.tellg();
    while(getline(input_file, line)){
        if(!line.empty()) ++num_line;
    }
    input_file.clear();
    input_file.seekg(pos);

    Eigen::MatrixXd X(num_line, 6);
    Eigen::VectorXd y(num_line);

    int row = 0;
    while (getline(input_file, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        string token;
        vector<double> values;
        
        // Split line by tab
        while (getline(iss, token, '\t')) {
            values.push_back(stod(token));
        }
        if (values.size() != 6) continue;

        // Fill row directly, cache-friendly
        X(row, 0) = 1.0;  // intercept
        for (int j = 1; j < 6; ++j) {
            X(row, j) = values[j - 1];  // features f1..f5
        }
        y(row) = values[5];  // output y

        ++row;
    }

    // Linear regression: β = (X^T X)^-1 X^T y
    Eigen::VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);

    cout << "\n Coefficients (beta):\n" << beta.transpose() << "\n";

    // Compute t-statistics in parallel
    Eigen::VectorXd t_stats = T_static(X, y, beta);

    cout << "\n t-statistics:\n" << t_stats.transpose() << "\n";

    // Significance test
    cout << "\n Coefficient significance:\n";

    int p = t_stats.size();
    for (int i = 0; i < p; ++i) {
        
        if (abs(t_stats(i)) < 2.0)
            cout << "beta[" << i << "] approx 0 (Not significant)\n";
        else
            cout << "beta[" << i << "] is significant\n";

    }

    return 0;
}

