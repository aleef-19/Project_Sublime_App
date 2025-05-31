#include <Rcpp.h>
#include <vector>
#include <thread>
#include <algorithm>

using namespace std;
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]

// Worker function for parallel blocked and unrolled matrix multiplication
void parallelBlockedUnrolledWorker(
    const NumericMatrix& A, const NumericMatrix& B, NumericMatrix& result,
    int blockSize, int rowStart, int rowEnd
) {
    int c1 = A.ncol();  // Number of columns in matrix A
    int c2 = B.ncol();  // Number of columns in matrix B

    // Iterate over blocks of rows
    for (int i0 = rowStart; i0 < rowEnd; i0 += blockSize) {
        int iMax = min(i0 + blockSize, rowEnd);

        // Iterate over blocks of columns
        for (int j0 = 0; j0 < c2; j0 += blockSize) {
            int jMax = min(j0 + blockSize, c2);

            // Iterate over blocks in the shared dimension
            for (int k0 = 0; k0 < c1; k0 += blockSize) {
                int kMax = min(k0 + blockSize, c1);

                // Process the block
                for (int i = i0; i < iMax; ++i) {
                    // Local buffer to accumulate results for current row and block of columns
                    vector<double> localResult(jMax - j0, 0.0);

                    for (int k = k0; k < kMax; ++k) {
                        double r = A(i, k);  // Element from A

                        // Loop unrolling: process 4 columns at a time
                        int j = 0;
                        for (; j + 3 < (jMax - j0); j += 4) {
                            localResult[j]     += r * B(k, j0 + j);
                            localResult[j + 1] += r * B(k, j0 + j + 1);
                            localResult[j + 2] += r * B(k, j0 + j + 2);
                            localResult[j + 3] += r * B(k, j0 + j + 3);
                        }

                        // Handle remaining columns
                        for (; j < (jMax - j0); ++j) {
                            localResult[j] += r * B(k, j0 + j);
                        }
                    }

                    // Store the computed local result into the global result matrix
                    for (int j = 0; j < (jMax - j0); ++j) {
                        result(i, j0 + j) += localResult[j];
                    }
                }
            }
        }
    }
}

// [[Rcpp::export]]
NumericMatrix parallelBlockedUnrolledMultiplyR(NumericMatrix A, NumericMatrix B, int blockSize = 128) {
    int r1 = A.nrow();    // Number of rows in matrix A
    int c2 = B.ncol();    // Number of columns in matrix B

    NumericMatrix result(r1, c2);  // Output matrix initialized to 0

    // Get number of hardware threads supported
    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;  // Fallback to 4 if not detectable

    vector<thread> threads;
    int rowsPerThread = (r1 + numThreads - 1) / numThreads;  // Divide work among threads

    // Launch threads
    for (int t = 0; t < numThreads; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = min(startRow + rowsPerThread, r1);
        if (startRow >= r1) break;

        threads.emplace_back(parallelBlockedUnrolledWorker,
                             cref(A), cref(B), ref(result),
                             blockSize, startRow, endRow);
    }

    // Wait for all threads to complete
    for (auto& th : threads) {
        th.join();
    }

    return result;  // Return the final multiplied matrix
}


