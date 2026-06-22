/**
 * Matrix Multiplication Example using GPU Acceleration
 * 
 * This program demonstrates how to multiply two large matrices using a GPU.
 * It shows:
 * 1. How to organize data for parallel processing
 * 2. How to transfer data between CPU and GPU
 * 3. How to divide work among many processing units
 * 4. How to measure performance
 * 
 * The program multiplies two NxN matrices (where N = 1024 by default)
 * and shows the result of the first 3x3 elements for verification.
 * 
 * Students can experiment with different floating-point types by changing
 * the real_t typedef at the top of the file.
 * 
 * To use half precision (__half):
 * 1. Change the typedef to: typedef __half real_t;
 * 2. Make sure your GPU supports half precision (compute capability 5.3 or higher)
 * 3. Compile with: nvcc -arch=sm_53 matrix_multiply.cu -o matrix_multiply
 */

// nvcc -o matrix_multiply matrix_multiply.cu
// ./matrix_multiply

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // Required for half precision support
#include <time.h>

// Type for our matrix elements - students can change this to experiment with different types
// Options include: 
// - float (single precision, 32-bit)
// - double (double precision, 64-bit)
// - __half (half precision, 16-bit, requires compute capability 5.3+)
typedef double real_t;  

// Define the size of our matrices and how we'll divide the work
#define MATRIX_SIZE 1024    // Size of each matrix (N x N)
#define BLOCK_SIZE 32      // Number of threads that work together (must be multiple of 32)

// Helper function to check for errors in GPU operations
// This makes debugging easier by showing exactly where any problems occur
#define cudaCheckError() {                                                              \
    cudaError_t e = cudaGetLastError();                                                 \
    if (e != cudaSuccess) {                                                             \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(EXIT_FAILURE);                                                             \
    }                                                                                   \
}

// Helper function to convert float to half precision
// Only needed if using __half type
#ifdef __CUDA_ARCH__
__device__ __half float2half(float x) {
    return __float2half(x);
}
#endif

// Helper function to convert half to float for printing
#ifdef __CUDA_ARCH__
__device__ float half2float(__half x) {
    return __half2float(x);
}
#endif

// Helper function to convert random number to real_t
// This handles the conversion properly for all types including __half
template<typename T>
T rand_to_real() {
    float temp = rand() / (float)RAND_MAX;
    return static_cast<T>(temp);
}

/**
 * The main computation function that runs on the GPU
 * This function is called a "kernel" and runs in parallel on many GPU cores
 * 
 * Parameters:
 * A: First input matrix
 * B: Second input matrix
 * C: Output matrix (result)
 * size: Size of the matrices (N)
 */
__global__ void matrixMultiply(real_t *A, real_t *B, real_t *C, int size) {
    // Each thread calculates one element of the result matrix
    // This line figures out which element this thread is responsible for
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we don't try to calculate elements outside our matrix
    if (row < size && col < size) {
        real_t sum = 0.0;

        // TODO: compute one output element as the dot product of row `row`
        //       of A with column `col` of B. The formula is
        //       C[i,j] = sum_k A[i,k] * B[k,j]:
        //   for (int i = 0; i < size; i++)
        //     sum += A[row * size + i] * B[i * size + col];

        // TODO: store the result back to global memory:
        //   C[row * size + col] = sum;
    }
}

/**
 * Helper function to measure time accurately
 * Returns current time in milliseconds
 */
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
}

int main() {
    // Set up our matrix dimensions and calculate memory needed
    const int size = MATRIX_SIZE;
    const int bytes = size * size * sizeof(real_t);
    
    // Calculate total number of floating-point operations
    // For matrix multiplication, each element requires:
    // - N multiplications
    // - N-1 additions
    // Total operations = N * N * (2N - 1) = 2N³ - N²
    const double total_flops = 2.0 * size * size * size - size * size;
    
    // Step 1: Prepare our data on the CPU
    // Allocate memory for our matrices
    real_t *h_A = (real_t*)malloc(bytes);  // First input matrix
    real_t *h_B = (real_t*)malloc(bytes);  // Second input matrix
    real_t *h_C = (real_t*)malloc(bytes);  // Result matrix
    
    // Fill our matrices with random numbers between 0 and 1
    for (int i = 0; i < size * size; i++) {
        h_A[i] = rand_to_real<real_t>();
        h_B[i] = rand_to_real<real_t>();
    }
    
    // Step 2: Set up GPU memory
    // Allocate memory on the GPU for our matrices
    real_t *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;  // 'd_' = GPU (device) memory
    // TODO: allocate d_A, d_B, d_C on the device, e.g.
    //   cudaMalloc((void**)&d_A, bytes);
    cudaCheckError();

    // Step 3: Copy data from CPU to GPU
    // This is necessary because GPU has its own separate memory
    // TODO: copy h_A -> d_A and h_B -> d_B, e.g.
    //   cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // Step 4: Plan how to divide the work
    // threadsPerBlock: How many threads work together (32x32 grid)
    // blocksPerGrid: How many blocks we need to cover the whole matrix
    // dim3 is a CUDA type that specifies a 3D grid/block size
    // Here we create a 2D block of BLOCK_SIZE x BLOCK_SIZE threads (e.g. 32x32)
    // This means each thread block will be a square of threads working together
    // TODO: create a 2D block of BLOCK_SIZE x BLOCK_SIZE threads:
    //   dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Calculate the number of blocks needed to cover the whole matrix
    // We add (BLOCK_SIZE-1) to size before dividing to round up to the next block
    // For example: if size=1000 and BLOCK_SIZE=32
    // Then (1000+31)/32 = 1031/32 = 32.21875 which rounds down to 32 blocks
    // This ensures we have enough blocks to cover all elements
    // TODO: create the grid, rounding up in each dimension:
    //   dim3 blocksPerGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE,
    //                      (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Step 5: Run the calculation and measure time
    printf("Performing matrix multiplication...\n");
    double start_time = get_time_ms();
    
    // Launch our calculation on the GPU
    // TODO: launch the kernel with your grid and block dimensions:
    //   matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
    cudaCheckError();
    
    // Wait for all GPU threads to finish
    cudaDeviceSynchronize();
    cudaCheckError();
    
    // Calculate how long the operation took
    double end_time = get_time_ms();
    double execution_time_ms = end_time - start_time;
    
    // Calculate performance metrics
    // Convert our measurements to FLOPS (Floating Point Operations Per Second)
    double flops_per_second = (total_flops / execution_time_ms) * 1000.0;  // Convert ms to seconds
    double gflops = flops_per_second / 1e9;  // Convert to billions of FLOPS (GFLOPS)
    
    // Step 6: Get our results back from the GPU
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    // Step 7: Show our results
    printf("\nPerformance Metrics:\n");
    printf("Matrix Size: %d x %d\n", size, size);
    printf("Total FLOPS: %.2e\n", total_flops);
    printf("Execution Time: %.2f ms\n", execution_time_ms);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // Step 8: Verify our results are correct
    // We check the first 3x3 elements by calculating them on the CPU
    printf("\nVerification (showing first 3x3 elements of result):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            real_t expected = 0.0;
            for (int k = 0; k < size; k++) {
                expected += h_A[i * size + k] * h_B[k * size + j];
            }
            printf("C[%d][%d] = %.2f (expected: %.2f)\n", i, j, h_C[i * size + j], expected);
        }
    }
    
    // Step 9: Clean up
    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("\nMatrix multiplication completed successfully.\n");
    return 0;
}