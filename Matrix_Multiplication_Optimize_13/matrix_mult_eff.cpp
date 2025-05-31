#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm> // for min
#include <cstdlib>   // for rand

using namespace std;
using namespace std::chrono;

// Technique 1: Naive multiplication (basic triple-loop matrix multiplication)
void naiveMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& result) {
    int r1 = A.size();
    int c1 = A[0].size();
    int c2 = B[0].size();
    result.assign(r1, vector<int>(c2, 0)); // initialize result matrix
    for (int i = 0; i < r1; ++i) {
        for (int j = 0; j < c2; ++j) {
            for (int k = 0; k < c1; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Technique 2: Loop unrolling (unroll inner loop by 4 for better performance)
void loopUnrolledMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& result) {
    int r1 = A.size();
    int c1 = A[0].size();
    int c2 = B[0].size();
    result.assign(r1, vector<int>(c2, 0));
    for (int i = 0; i < r1; ++i) {
        for (int j = 0; j < c2; ++j) {
            int sum = 0;
            int k = 0;
            // unrolled loop processing 4 elements at a time
            for (; k + 3 < c1; k += 4) {
                sum += A[i][k] * B[k][j]
                    + A[i][k+1] * B[k+1][j]
                    + A[i][k+2] * B[k+2][j]
                    + A[i][k+3] * B[k+3][j];
            }
            // process any remaining elements
            for (; k < c1; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
}

// Technique 3: Blocked matrix multiplication (cache friendly)
void blockedMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& result, int blockSize = 64) {
    int r1 = A.size();
    int c1 = A[0].size();
    int c2 = B[0].size();
    result.assign(r1, vector<int>(c2, 0));
    
    // block-based iteration to improve cache locality
    for (int i0 = 0; i0 < r1; i0 += blockSize) {
        for (int j0 = 0; j0 < c2; j0 += blockSize) {
            for (int k0 = 0; k0 < c1; k0 += blockSize) {
                for (int i = i0; i < min(i0 + blockSize, r1); ++i) {
                    for (int k = k0; k < min(k0 + blockSize, c1); ++k) {
                        int r = A[i][k];
                        for (int j = j0; j < min(j0 + blockSize, c2); ++j) {
                            result[i][j] += r * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

// Worker function for parallel matrix multiplication (each thread processes a subset of rows)
void parallelMultiplyWorker(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& result, int startRow, int endRow) {
    int c1 = A[0].size();
    int c2 = B[0].size();
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < c2; ++j) {
            int sum = 0;
            for (int k = 0; k < c1; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
}

// Technique 4: Parallel computing (divides the workload across multiple threads)
void parallelMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& result) {
    int r1 = A.size();
    int c2 = B[0].size();
    result.assign(r1, vector<int>(c2, 0));
    int numThreads = thread::hardware_concurrency(); // get number of available cores
    if (numThreads == 0) numThreads = 4; // fallback
    vector<thread> threads;
    int rowsPerThread = (r1 + numThreads - 1) / numThreads;

    for (int t = 0; t < numThreads; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = min(startRow + rowsPerThread, r1);
        if (startRow >= r1) break;
        threads.emplace_back(parallelMultiplyWorker, cref(A), cref(B), ref(result), startRow, endRow);
    }
    for (auto& th : threads) th.join(); // wait for all threads to finish
}

// Worker function for advanced matrix multiplication using blocking, unrolling, and parallelism
void parallelBlockedUnrolledWorkerImproved(
    const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& result,
    int blockSize, int rowStart, int rowEnd)
{
    int c1 = A[0].size();
    int c2 = B[0].size();

    for (int i0 = rowStart; i0 < rowEnd; i0 += blockSize) {
        int iMax = min(i0 + blockSize, rowEnd);
        for (int j0 = 0; j0 < c2; j0 += blockSize) {
            int jMax = min(j0 + blockSize, c2);
            for (int k0 = 0; k0 < c1; k0 += blockSize) {
                int kMax = min(k0 + blockSize, c1);

                for (int i = i0; i < iMax; ++i) {
                    vector<int> localResult(jMax - j0, 0); // local buffer for improved cache use

                    for (int k = k0; k < kMax; ++k) {
                        int r = A[i][k];

                        int j = 0;
                        // unrolled loop for performance
                        for (; j + 3 < (jMax - j0); j += 4) {
                            localResult[j]     += r * B[k][j0 + j];
                            localResult[j + 1] += r * B[k][j0 + j + 1];
                            localResult[j + 2] += r * B[k][j0 + j + 2];
                            localResult[j + 3] += r * B[k][j0 + j + 3];
                        }
                        for (; j < (jMax - j0); ++j) {
                            localResult[j] += r * B[k][j0 + j];
                        }
                    }
                    // store result from local buffer
                    for (int j = 0; j < (jMax - j0); ++j) {
                        result[i][j0 + j] += localResult[j];
                    }
                }
            }
        }
    }
}

// Technique 5: Combination of parallel, blocking, and unrolling techniques for high performance
void parallelBlockedUnrolledMultiplyImproved(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& result, int blockSize = 128) {
    int r1 = A.size();
    int c2 = B[0].size();
    result.assign(r1, vector<int>(c2, 0));
    int numThreads = thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    vector<thread> threads;
    int rowsPerThread = (r1 + numThreads - 1) / numThreads;

    for (int t = 0; t < numThreads; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = min(startRow + rowsPerThread, r1);
        if (startRow >= r1) break;
        threads.emplace_back(parallelBlockedUnrolledWorkerImproved, cref(A), cref(B), ref(result), blockSize, startRow, endRow);
    }
    for (auto& th : threads) th.join();
}

// Utility function to time any of the above multiplication techniques
template<typename Func>
double timeFunction(Func f, const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& result) {
    auto start = high_resolution_clock::now();
    f(A, B, result);
    auto stop = high_resolution_clock::now();
    return duration<double>(stop - start).count();
}

// Generate random square matrix
vector<vector<int>> generateMatrix(int n, int maxVal = 10) {
    vector<vector<int>> mat(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            mat[i][j] = rand() % maxVal;
    return mat;
}

int main() {
    int n = 512;  // Matrix size
    vector<vector<int>> A = generateMatrix(n);
    vector<vector<int>> B = generateMatrix(n);
    vector<vector<int>> result;

    cout << "Timing Naive Multiply...\n";
    double naiveTime = timeFunction(naiveMultiply, A, B, result);
    cout << "Naive: " << naiveTime << " seconds\n\n";

    cout << "Timing Loop Unrolled Multiply...\n";
    double unrolledTime = timeFunction(loopUnrolledMultiply, A, B, result);
    cout << "Loop Unrolled: " << unrolledTime << " seconds\n\n";

    cout << "Timing Blocked Multiply...\n";
    double blockedTime = timeFunction([](const auto& A, const auto& B, auto& result) {
        blockedMultiply(A, B, result, 64);
    }, A, B, result);
    cout << "Blocked: " << blockedTime << " seconds\n\n";

    cout << "Timing Parallel Multiply...\n";
    double parallelTime = timeFunction(parallelMultiply, A, B, result);
    cout << "Parallel: " << parallelTime << " seconds\n\n";

    cout << "Timing Improved Parallel + Blocked + Unrolled Multiply...\n";
    double improvedCombinedTime = timeFunction([](const auto& A, const auto& B, auto& result) {
        parallelBlockedUnrolledMultiplyImproved(A, B, result, 128);
    }, A, B, result);
    cout << "Improved Combined: " << improvedCombinedTime << " seconds\n\n";

    // Print speedups for performance comparison
    cout << "Speedups:\n";
    cout << "Loop Unrolled vs Naive: " << naiveTime / unrolledTime << "x faster\n";
    cout << "Blocked vs Naive: " << naiveTime / blockedTime << "x faster\n";
    cout << "Parallel vs Naive: " << naiveTime / parallelTime << "x faster\n";
    cout << "Improved Combined vs Naive: " << naiveTime / improvedCombinedTime << "x faster\n";

    return 0;
}




