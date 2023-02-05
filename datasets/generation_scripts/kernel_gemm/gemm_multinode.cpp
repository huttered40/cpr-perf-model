#include <fstream>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include "mkl.h"

int main( int argc, char** argv )
{
    int min_iter=5;// This must be set by user to generate a distinct binary
    int max_iter=20;// This must be set by user to generate a distinct binary
    double rel_std_dev_tol=0.01;// This must be set by user to generate a distinct binary
    std::string file_location = "/work2/05608/tg849075/cpr-perf-model/datasets/generation_scripts/kernel_gemm/gemm";

    double delta,msq,mean,std_dev,rel_std_dev;
    int kernel_type = atoi(argv[1]);
    int sample_type = atoi(argv[2]);
    int m = atoi(argv[3]);
    int n = atoi(argv[4]);
    int k = atoi(argv[5]);
    int thread_count = atoi(argv[6]);
    int id = atoi(argv[7]);
    int pid = atoi(argv[8]);
    std::ofstream write_file;
    std::string write_str = file_location + "_kt"
      +std::to_string(kernel_type)+"_st"+std::to_string(sample_type)
      +"_"+std::to_string(thread_count)+"threads_pid"+std::to_string(pid)+".csv";
    write_file.open(write_str,std::fstream::app);//,std::ios_base::app);

    // Generate matrices
    std::vector<double> A(m*k,1.);
    std::vector<double> B(k*n,1.);
    std::vector<double> C(m*n,1.);
    // Warm-up with num_iter iterations
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.,&A[0],m,&B[0],k,0.,&C[0],m);
    mean=0.; msq=0; delta=0;
    int j=0;
    while (1){
      auto start_time = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.,&A[0],m,&B[0],k,0.,&C[0],m);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end - start_time;
      delta = diff.count() - mean; mean += delta/(j+1); msq += delta*(diff.count()-mean);
      ++j;
      std_dev = j>1 ? sqrt(msq/(j-1)) : 100000.; rel_std_dev = std_dev/mean;
      if (j>= min_iter && rel_std_dev < rel_std_dev_tol) break; else if (j>=max_iter) break;
    }
    write_file << id << "," << m << "," << n << "," << k << "," << mean << "," << rel_std_dev << std::endl;
   return 0;
}
