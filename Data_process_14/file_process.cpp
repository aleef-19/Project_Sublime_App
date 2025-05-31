#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <omp.h>  // OpenMP for parallelization

struct Record {
    double height;
    double weight;
};

// Split CSV line by comma into two doubles
bool parse_line(const std::string& line, Record& rec) {
    std::stringstream ss(line);
    std::string token;
    
    if (!std::getline(ss, token, ',')) return false;
    try {
        rec.height = std::stod(token);
    } catch (...) { return false; }
    
    if (!std::getline(ss, token, ',')) return false;
    try {
        rec.weight = std::stod(token);
    } catch (...) { return false; }
    
    return true;
}

int main() {
    const std::string filename = "height_weight_large.csv";
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file.\n";
        return 1;
    }

    std::string line;
    std::vector<std::string> lines;

    // Read header line and ignore it
    std::getline(file, line);

    // Read all lines and store in vector
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    // Print lines 2025 and 3106 (1-based indexing)
    if (lines.size() >= 3106) {
        std::cout << "Line 2025: " << lines[2024] << "\n";
        std::cout << "Line 3106: " << lines[3105] << "\n";
    } else {
        std::cerr << "File has fewer lines than required.\n";
    }

    // Prepare container for parsed data
    std::vector<Record> records(lines.size());

    // Parallel parse using OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < lines.size(); ++i) {
        if (!parse_line(lines[i], records[i])) {
            // Handle parse error, if needed
            records[i].height = 0;
            records[i].weight = 0;
        }
    }

    return 0;
}


