#include <fstream>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
//#include "/work2/05608/tg849075/openmpi-4.1.4/_install/include/mpi.h"
#include <mpi.h>
//#include "mkl.h"

int main( int argc, char** argv )
{
    int min_iter=5;// This must be set by user to generate a distinct binary
    int max_iter=50;// This must be set by user to generate a distinct binary
    double rel_std_dev_tol=0.01;// This must be set by user to generate a distinct binary
    std::string file_location = "/work2/05608/tg849075/cpr-perf-model/datasets/generation_scripts/kernel_bcast/data/bcast";

    int mpi_rank,mpi_size;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    double delta,msq,mean,std_dev,rel_std_dev;
    int kernel_type = atoi(argv[1]);
    int sample_type = atoi(argv[2]);
    int message_size = atoi(argv[3]);
    int ppn = atoi(argv[4]);
    int num_nodes = atoi(argv[5]);
    int id = atoi(argv[6]);
    std::ofstream write_file;
    std::string write_str = file_location + "_kt"
      +std::to_string(kernel_type)+"_st"+std::to_string(sample_type)
      +"_nodes"+std::to_string(num_nodes)+"_ppn"+std::to_string(ppn)+".csv";
    if (mpi_rank == 0){
      write_file.open(write_str,std::fstream::app);//,std::ios_base::app);
    }

    // Generate message
    std::vector<char> A(message_size,1.);
    // Warm-up with num_iter iterations
    MPI_Bcast(&A[0],message_size,MPI_CHAR,0,MPI_COMM_WORLD);
    mean=0.; msq=0; delta=0;
    int j=0;
    while (1){
      auto start_time = std::chrono::high_resolution_clock::now();
      MPI_Bcast(&A[0],message_size,MPI_CHAR,0,MPI_COMM_WORLD);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end - start_time;
      delta = diff.count() - mean; mean += delta/(j+1); msq += delta*(diff.count()-mean);
      ++j;
      std_dev = j>1 ? sqrt(msq/(j-1)) : 100000.; rel_std_dev = std_dev/mean;
      /*if (j>= min_iter && rel_std_dev < rel_std_dev_tol) break; else*/ if (j>=max_iter) break;
    }
    if (mpi_rank == 0){
      write_file << id << "," << message_size << "," << ppn << "," << num_nodes << "," << mean << "," << rel_std_dev << std::endl;
    }
   MPI_Finalize();
   return 0;
}
