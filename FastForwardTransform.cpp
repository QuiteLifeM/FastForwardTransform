#include <iostream>
#include <vector>
#include <complex>
#include <cstdlib> 
#include <cmath>   
#include <fftw3.h>

class FFTWTransform {
public:
    FFTWTransform(int N) : N(N) {
        in = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
        out = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
        forward_plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        backward_plan = fftw_plan_dft_1d(N, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    ~FFTWTransform() {
        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(backward_plan);
        fftw_free(in);
        fftw_free(out);
    }

    std::vector<std::complex<double>> forwardTransform(const std::vector<std::complex<double>>& input) {
        std::vector<std::complex<double>> output(N);

        for (int i = 0; i < N; ++i) {
            in[i][0] = input[i].real();
            in[i][1] = input[i].imag();
        }

        fftw_execute(forward_plan);

        for (int i = 0; i < N; ++i) {
            output[i] = std::complex<double>(out[i][0], out[i][1]);
        }

        return output;
    }

    std::vector<std::complex<double>> inverseTransform(const std::vector<std::complex<double>>& input) {
        std::vector<std::complex<double>> output(N);

        for (int i = 0; i < N; ++i) {
            out[i][0] = input[i].real();
            out[i][1] = input[i].imag();
        }

        fftw_execute(backward_plan);

        for (int i = 0; i < N; ++i) {
            output[i] = std::complex<double>(in[i][0] / N, in[i][1] / N);
        }

        return output;
    }

private:
    int N;
    fftw_complex* in;
    fftw_complex* out;
    fftw_plan forward_plan;
    fftw_plan backward_plan;
};

int main() {
    const int TWO_MULTIPLIER = 2;
    const int THREE_MULTIPLIER = 3;
    const int FIVE_MULTIPLIER = 5;

    int inputLength;
    std::cout << "Enter the input length (should be a multiple of 2, 3, or 5): ";
    std::cin >> inputLength;

    if (inputLength % TWO_MULTIPLIER != 0 && inputLength % THREE_MULTIPLIER != 0 && inputLength % FIVE_MULTIPLIER != 0) {
        std::cerr << "Input length must be a multiple of 2, 3, or 5." << std::endl;
        return 1;
    }

    FFTWTransform fft(inputLength);

    std::vector<std::complex<double>> input;
    for (int i = 0; i < inputLength; ++i) {
        double real = static_cast<double>(rand()) / RAND_MAX;
        double imag = static_cast<double>(rand()) / RAND_MAX;
        input.emplace_back(real, imag);
    }

    std::vector<std::complex<double>> spectrum = fft.forwardTransform(input);

    std::vector<std::complex<double>> reconstructed = fft.inverseTransform(spectrum);

    double error = 0.0;
    for (int i = 0; i < inputLength; ++i) {
        error += std::norm(input[i] - reconstructed[i]);
    }
    error = std::sqrt(error / inputLength);

    std::cout << "Error between input and output data: " << error << std::endl;

    return 0;
}
