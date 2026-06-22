#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>

constexpr int NumPatients = 1000000;  // Number of patients
constexpr int NumFeatures = 10;       // Number of health features

// Simple timer class for measuring execution time
class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::string function_name;

public:
    // Constructor starts the timer
    Timer(const std::string& name) : function_name(name) {
        start_time = std::chrono::high_resolution_clock::now();
        std::cout << "Starting " << function_name << "...\n";
    }

    // Destructor automatically stops the timer and prints elapsed time
    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << function_name << " completed in " << duration.count() << " ms\n";
    }
};

// Function to generate synthetic health data
void generate_data(std::vector<std::vector<double>>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(50, 150);  // Values between 50 and 150

    #pragma omp parallel for
    for (int i = 0; i < NumPatients; i++) {
        for (int j = 0; j < NumFeatures; j++) {
            data[i][j] = dist(gen);
        }
    }
}

// Compute mean of each feature
void compute_mean(const std::vector<std::vector<double>>& data, std::vector<double>& mean) {
    #pragma omp parallel for
    for (int j = 0; j < NumFeatures; j++) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < NumPatients; i++) {
            sum += data[i][j];
        }
        mean[j] = sum / NumPatients;
    }
}

// Compute variance of each feature
void compute_variance(const std::vector<std::vector<double>>& data, const std::vector<double>& mean, std::vector<double>& variance) {
    #pragma omp parallel for
    for (int j = 0; j < NumFeatures; j++) {
        double sum_sq = 0.0;
        #pragma omp parallel for reduction(+:sum_sq)
        for (int i = 0; i < NumPatients; i++) {
            sum_sq += std::pow(data[i][j] - mean[j], 2);
        }
        variance[j] = sum_sq / (NumPatients - 1);  // Sample variance
    }
}

// Compute correlation matrix
void compute_correlation(const std::vector<std::vector<double>>& data, const std::vector<double>& mean, const std::vector<double>& variance, std::vector<std::vector<double>>& correlation) {
    #pragma omp parallel for collapse(2)
    for (int j1 = 0; j1 < NumFeatures; j1++) {
        for (int j2 = 0; j2 < NumFeatures; j2++) {
            if (j1 == j2) {
                correlation[j1][j2] = 1.0;  // Correlation with itself is always 1
                continue;
            }
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < NumPatients; i++) {
                sum += (data[i][j1] - mean[j1]) * (data[i][j2] - mean[j2]);
            }
            correlation[j1][j2] = sum / ((NumPatients - 1) * std::sqrt(variance[j1]) * std::sqrt(variance[j2]));
        }
    }
}

// Main function
int main() {
    std::cout << "Analyzing health data for " << NumPatients << " patients with " << NumFeatures << " features\n";
    int nthreads = omp_get_max_threads();

    std::cout << "Number of threads: " << nthreads << std::endl;
    {
        // Create timer for total runtime
        Timer timer("Total Runtime");

        std::vector<std::vector<double>> data(NumPatients, std::vector<double>(NumFeatures));
        std::vector<double> mean(NumFeatures, 0.0), variance(NumFeatures, 0.0);
        std::vector<std::vector<double>> correlation(NumFeatures, std::vector<double>(NumFeatures, 0.0));
        


        {
            Timer timer("Generate Data");
            generate_data(data);
        }
        
        {
            Timer timer("Compute Mean");
            compute_mean(data, mean);
        }
        
        {
            Timer timer("Compute Variance");
            compute_variance(data, mean, variance);
        }
        
        {
            Timer timer("Compute Correlation");
            compute_correlation(data, mean, variance, correlation);
        }

        std::cout << "\nHealth Feature Summary:\n";
        for (int j = 0; j < NumFeatures; j++) {
            std::cout << "Feature " << j << " - Mean: " << mean[j] << ", Variance: " << variance[j] << "\n";
        }

        std::cout << "\nCorrelation Matrix (first 3x3 section for brevity):\n";
        for (int j1 = 0; j1 < 3; j1++) {
            for (int j2 = 0; j2 < 3; j2++) {
                std::cout << correlation[j1][j2] << " ";
            }
            std::cout << "\n";
        }
    }

    return 0;
}