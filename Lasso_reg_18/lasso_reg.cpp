#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

typedef vector<double> Vec;
typedef vector<Vec> Mat;

void readCSV(const string& filename, Mat &X, Vec &y) {
    ifstream file(filename);
    string line;
    getline(file, line); // skip header
    
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        Vec row;
        int col_count = 0;
        while (getline(ss, cell, ',')) {
            double val = stod(cell);
            if (col_count < 10) row.push_back(val);
            else y.push_back(val);
            col_count++;
        }
        if (!row.empty()) X.push_back(row);
    }
}

void standardize(Mat &X, Vec &mean, Vec &stddev) {
    int n = X.size();
    int p = X[0].size();
    mean.resize(p, 0.0);
    stddev.resize(p, 0.0);

    // Compute mean
    for (int j = 0; j < p; j++) {
        for (int i = 0; i < n; i++) mean[j] += X[i][j];
        mean[j] /= n;
    }
    // Compute stddev
    for (int j = 0; j < p; j++) {
        for (int i = 0; i < n; i++) {
            double diff = X[i][j] - mean[j];
            stddev[j] += diff * diff;
        }
        stddev[j] = sqrt(stddev[j] / n);
        if (stddev[j] == 0) stddev[j] = 1; // avoid divide by zero
    }
    // Standardize X
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            X[i][j] = (X[i][j] - mean[j]) / stddev[j];
        }
    }
}

Vec lassoCoordinateDescent(const Mat &X, const Vec &y, double lambda, int max_iter=1000, double tol=1e-6) {
    int n = X.size();
    int p = X[0].size();
    Vec beta(p, 0.0);
    Vec residual = y; // initial residual = y - X*beta = y

    for (int iter = 0; iter < max_iter; iter++) {
        Vec beta_old = beta;

        for (int j = 0; j < p; j++) {
            // Compute partial residual excluding feature j
            double rho = 0.0;
            for (int i = 0; i < n; i++) {
                double pred_except_j = 0.0;
                // residual[i] + beta[j]*X[i][j] = y[i] - sum_{k != j} beta[k]*X[i][k]
                // so rho_j = sum_i X[i][j] * (y[i] - sum_{k != j} beta[k] X[i][k]) = sum_i X[i][j]*(residual[i] + beta[j]*X[i][j])
                // But we keep residual = y - X*beta, so we update residual for coordinate descent
                
                // So we use residual[i] + beta[j]*X[i][j]
                rho += X[i][j] * (residual[i] + beta[j] * X[i][j]);
            }
            // Update beta_j with soft thresholding
            double z = 0.0;
            for (int i = 0; i < n; i++) z += X[i][j] * X[i][j];
            double bj = 0.0;
            if (rho < -lambda) bj = (rho + lambda) / z;
            else if (rho > lambda) bj = (rho - lambda) / z;
            else bj = 0.0;

            // Update residual vector for next iteration
            for (int i = 0; i < n; i++) {
                residual[i] += (beta[j] - bj) * X[i][j];
            }
            beta[j] = bj;
        }
        // Check convergence
        double max_change = 0.0;
        for (int j = 0; j < p; j++) max_change = max(max_change, fabs(beta[j] - beta_old[j]));
        if (max_change < tol) break;
    }
    return beta;
}

double computeRMSE(const Mat &X, const Vec &y, const Vec &beta) {
    int n = X.size();
    int p = X[0].size();
    double error_sum = 0.0;
    for (int i = 0; i < n; i++) {
        double pred = 0.0;
        for (int j = 0; j < p; j++) pred += X[i][j] * beta[j];
        double err = y[i] - pred;
        error_sum += err * err;
    }
    return sqrt(error_sum / n);
}

int main() {
    Mat X;
    Vec y;

    readCSV("lasso_data.dat", X, y);
    cout << "Read " << X.size() << " rows and " << X[0].size() << " features." << endl;

    // Standardize features
    Vec mean, stddev;
    standardize(X, mean, stddev);

    // Grid search for lambda (adjust grid as needed)
    vector<double> lambdas = {0.01, 0.1, 1, 10, 100};
    double best_lambda = lambdas[0];
    Vec best_beta;
    double best_rmse = numeric_limits<double>::max();

    for (double lambda : lambdas) {
        Vec beta = lassoCoordinateDescent(X, y, lambda);
        double rmse = computeRMSE(X, y, beta);
        cout << "Lambda: " << lambda << ", RMSE: " << rmse << endl;

        if (rmse < best_rmse) {
            best_rmse = rmse;
            best_beta = beta;
            best_lambda = lambda;
        }
    }

    cout << "\nBest lambda: " << best_lambda << endl;
    cout << "Best coefficients:" << endl;
    for (int j = 0; j < (int)best_beta.size(); j++) {
        cout << "beta[" << j+1 << "] = " << best_beta[j] << endl;
    }
    cout << "Best RMSE: " << best_rmse << endl;

    return 0;
}


