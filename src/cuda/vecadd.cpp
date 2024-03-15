

#include <iostream>

void print_vector(float *vector) {
    for (int i = 0; i < 2; ++i) {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}


void print_all(float *a, float *b, float *c) {
        std::cout << "a: ";
        print_vector(a);
        std::cout << "b: ";
        print_vector(b);
        std::cout << "c: ";
        print_vector(c);
}


extern "C" {
    void add_vectors_cuda(float *a, float *b, float *c);
    float add_vectors_cpp(float *a, float *b, float *c) {
        print_all(a, b, c);
        add_vectors_cuda(a, b, c);
        print_all(a, b, c);
    }
}

