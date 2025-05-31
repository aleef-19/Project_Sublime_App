# Load Rcpp package
library(Rcpp)

# Compile and load the optimized C++ matrix multiplication code
# Adjust the path below to point to your actual C++ file location
sourceCpp("C:/Users/kiros/OneDrive/Desktop/c_lang/matrix_mult.cpp")

# Set the size of the matrices
n <- 512

# Set a random seed for reproducibility
set.seed(123)

# Generate random matrices A and B with values between 0 and 1
A <- matrix(runif(n * n), n, n)
B <- matrix(runif(n * n), n, n)

# Call the optimized matrix multiplication function from C++
result <- parallelBlockedUnrolledMultiplyR(A, B, blockSize = 128)

# Display the first 5x5 block of the resulting matrix
cat("First 5x5 block of the result matrix:\n")
print(result[1:5, 1:5])

