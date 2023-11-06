#include <cstring>
#include <utility>
#include <algorithm>
#include <map>
#include <set>
#include <time.h>
#include <sys/time.h>

#include "cp_model.h"
#include "ctf.hpp"
/*
#include "interface/common.h"
#include "interface/world.h"
#include "interface/tensor.h"
*/

#define NUMERICAL_PARAM_MIN_OBS_RANGE 32
#define RE_INIT_LOSS_TOL 0.25		// If loss is higher than this, re-try optimization
#define MIN_NUM_OBS_FOR_TRAINING 128	// can override with environment variable CPPM_MIN_NUM_OBS_FOR_TRAINING
#define MIN_POS_RUNTIME 1e-8		// Essentially a "zero" runtime
#define MAX_NUM_RE_INITS 10
#define INTERPOLATION_FACTOR_TOL 0.5	// If difference in execution time between two nehboring tensor elements is larger than this factor, then interpolation shouldn't be used.

static double get_wall_time(){
  struct timeval time;
  if (gettimeofday(&time,NULL)){
    // Error
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
          
double multilinear_product(const double* multi_vector, const int* cells, const int* mode_lengths, int order, int rank){
  double tensor_elem_prediction = 0.;
  for (int j=0; j<rank; j++){
    double temp_val = 1;
    int offset = 0;
    for (int k=0; k<order; k++){
      temp_val *= multi_vector[offset+cells[k]*rank+j];
      offset += mode_lengths[k]*rank;
    }
    tensor_elem_prediction += temp_val;
  }
  return tensor_elem_prediction;
}

double multilinear_product_packed(const double* multi_vector, int nelements, int order, int rank){
  double t_val = 0;
  for (int l=0; l<rank; l++){
    double intermediate_term = 1;
    for (int ll=0; ll<order; ll++){
      assert(l+ll*rank < nelements);
      intermediate_term *= multi_vector[l+ll*rank];
    }
    t_val += intermediate_term;
  }
  return t_val;
}

void init_factor_matrix(CTF::Tensor<>* fm, const char* loss_function_str){
  if (fm==nullptr) return;
  //assert(fm != nullptr);
  int64_t num_nnz_elems;
  int64_t* ptr_to_indices;
  double* ptr_to_data;
  int cp_rank = fm->lens[0];
  int mode_length = fm->lens[1];
  //TODO: Use strcpy instead
  if (std::string(loss_function_str) == "MSE") fm->fill_random(-1,1);
  else if (std::string(loss_function_str) == "MLogQ2") fm->fill_random(0,.01);
  else assert(0);
  // Enforce to be strictly increasing
  fm->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,true);
  for (int k=0; k<cp_rank; k++){
    for (int j=1; j<mode_length; j++){
      ptr_to_data[j*cp_rank+k] += ptr_to_data[(j-1)*cp_rank+k];
    }
  }
  fm->write(num_nnz_elems,ptr_to_indices,ptr_to_data);
  delete[] ptr_to_data;
  delete[] ptr_to_indices;
}

bool aggregate_observations(int& num_configurations, const double*& configurations, const double*& runtimes, int order, MPI_Comm cm){
  // Return value indicates whether configurations and runtimes need to be explicitly deleted
  int rank,size,lnc;
  int internal_tag=0;
  int internal_tag1=0;
  int internal_tag2=0;
  MPI_Comm_rank(cm,&rank);
  MPI_Comm_size(cm,&size);
  if (size==1) return false;
  size_t active_size = size;
  size_t active_rank = rank;
  size_t active_mult = 1;
  int local_num_configurations = num_configurations;
  std::vector<double> local_configurations(num_configurations*order);
  std::vector<double> local_runtimes(num_configurations);
  for (int i=0; i<num_configurations; i++){
    for (int j=0; j<order; j++){
      local_configurations[i*order+j]=configurations[i*order+j];
    }
    local_runtimes[i]=runtimes[i];
  }
  while (active_size>1){
    if (active_rank % 2 == 1){
      int partner = (active_rank-1)*active_mult;
      // Send sizes before true message so that receiver can be aware of the array sizes for subsequent communication
      PMPI_Send(&local_num_configurations,1,MPI_INT,partner,internal_tag,cm);
      if (local_num_configurations>0){
        // Send active kernels with keys
        PMPI_Send(&local_configurations[0],local_num_configurations*order,MPI_DOUBLE,partner,internal_tag1,cm);
        PMPI_Send(&local_runtimes[0],local_num_configurations,MPI_DOUBLE,partner,internal_tag2,cm);
      }
      break;// important. Senders must not update {active_size,active_rank,active_mult}
    }
    else if ((active_rank % 2 == 0) && (active_rank < (active_size-1))){
      int partner = (active_rank+1)*active_mult;
      // Recv sizes of arrays to create buffers for subsequent communication. Goal is to concatenate, not to replace
      PMPI_Recv(&lnc,1,MPI_INT,partner,internal_tag,cm,MPI_STATUS_IGNORE);
      if (lnc>0){
        local_num_configurations += lnc;
        local_configurations.resize(local_num_configurations*order);
        local_runtimes.resize(local_num_configurations);
        PMPI_Recv(&local_configurations[order*(local_num_configurations-lnc)],lnc*order,MPI_DOUBLE,partner,internal_tag1,cm,MPI_STATUS_IGNORE);
        PMPI_Recv(&local_runtimes[local_num_configurations-lnc],lnc,MPI_DOUBLE,partner,internal_tag2,cm,MPI_STATUS_IGNORE);
      }
    }
    active_size = active_size/2 + active_size%2;
    active_rank /= 2;
    active_mult *= 2;
  }
  //if (rank==0) std::cout << "Total number of configurations (not necessarily distinct) after aggregation: " << local_num_configurations << ", yet " << num_configurations << " before\n";
  // Goal is to replace, not to concatenate
  PMPI_Bcast(&local_num_configurations,1,MPI_INT,0,cm);
  assert(num_configurations <= local_num_configurations);
  if (local_num_configurations == 0){
    return false;
  }
  if (rank != 0){
    local_configurations.resize(local_num_configurations*order);
    local_runtimes.resize(local_num_configurations);
  }
  PMPI_Bcast(&local_configurations[0],local_num_configurations*order,MPI_DOUBLE,0,cm);
  PMPI_Bcast(&local_runtimes[0],local_num_configurations,MPI_DOUBLE,0,cm);
  double* _runtimes = new double[local_num_configurations];
  double* _configurations = new double[local_num_configurations*order];
  for (int i=0; i<local_num_configurations; i++){
    for (int j=0; j<order; j++){
      _configurations[i*order+j]=local_configurations[i*order+j];
    }
    _runtimes[i]=local_runtimes[i];
  }
  configurations = _configurations;
  runtimes = _runtimes;
  num_configurations=local_num_configurations;
  return true;
}

void debug_tensor(CTF::Tensor<>* t, const std::string& str){
/*
  // Debug
  int64_t num_nnz_elems;
  int64_t* ptr_to_indices;
  double* ptr_to_data;
  t->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,false);
  std::cout << str << "  ";
  for (int j=0; j<num_nnz_elems; j++) std::cout << ptr_to_data[j] << " ";
  std::cout << "\n";
  delete[] ptr_to_data;
  delete[] ptr_to_indices;
  // End Debug
*/
}

// Normalize each column within a factor matrix
void normalize(std::vector<CTF::Tensor<>*>& X){
  // Re-normalize by iterating over each column of each factor matrix
  int64_t num_nnz_elems;
  int64_t* ptr_to_indices;
  double* ptr_to_data;
  int order = X.size();
  int rank = X[0]->lens[0];// choice of index 0 is arbitrary
  for (int j=0; j<rank; j++){
    double weight = 1;
    // Iterate over the j'th column of all d factor matrices
    for (int k=0; k<order; k++){
      X[k]->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,false);
      double nrm = 0;
      for (int l=0; l<X[k]->lens[1]; l++){
        double tmp_val = ptr_to_data[j + X[k]->lens[0]*l];
        nrm += tmp_val*tmp_val;
      }
      nrm = sqrt(nrm);
      weight *= nrm;
      for (int l=0; l<X[k]->lens[1]; l++){
        ptr_to_data[j + X[k]->lens[0]*l] /= nrm;
      }
      X[k]->write(num_nnz_elems,ptr_to_indices,ptr_to_data);
      delete[] ptr_to_data;
      delete[] ptr_to_indices;
    }
    weight = pow(weight,1./order);
    for (int k=0; k<order; k++){
      X[k]->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,false);
      for (int l=0; l<X[k]->lens[1]; l++){
        ptr_to_data[j + X[k]->lens[0]*l] *= weight;
      }
      X[k]->write(num_nnz_elems,ptr_to_indices,ptr_to_data);
      delete[] ptr_to_data;
      delete[] ptr_to_indices;
    }
  }
}

struct MLogQ2{
    //Current implementation is using \lambda  = e^m and replacing it in the function to get: e^m - xm
    static void Get_RHS(CTF::World* dw, CTF::Tensor<>* T, CTF::Tensor<>* O, std::vector<CTF::Tensor<>*>& A, int num, double reg, double mu, CTF::Tensor<>* grad, CTF::Tensor<>* Hessian){
        int64_t num_nnz_elems,num_nnz_elems2;
        int64_t* ptr_to_indices,*ptr_to_indices2;
        double* ptr_to_data,*ptr_to_data2;
        CTF::Tensor<> M(O);
        std::vector<int> mode_list(A.size());
        for (int j=0; j<A.size(); j++) mode_list[j]=j;
        CTF::TTTP(&M,A.size(),&mode_list[0],&A[0],true);
        auto M_reciprocal1 = M;
        auto M_reciprocal2 = M;

        M.get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,true);
        std::vector<double> new_data(num_nnz_elems); for (int i=0; i<num_nnz_elems; i++) new_data[i] = -1./ptr_to_data[i];
        M_reciprocal1.write(num_nnz_elems,ptr_to_indices,&new_data[0]);
        std::vector<double> new_data2(num_nnz_elems); for (int i=0; i<num_nnz_elems; i++) new_data2[i] = 1./(ptr_to_data[i]*ptr_to_data[i]);
        M_reciprocal2.write(num_nnz_elems,ptr_to_indices,&new_data2[0]);
        // Confirmed that swapping the negatives between new_data and new_data2 fails.
        double* ptr_to_data_t;
        T->get_local_data(&num_nnz_elems2,&ptr_to_indices2,&ptr_to_data_t,true);
        assert(num_nnz_elems2 == num_nnz_elems);
        for (int i=0; i<num_nnz_elems; i++) ptr_to_data[i] = log(ptr_to_data_t[i] / ptr_to_data[i]);
        M.write(num_nnz_elems,ptr_to_indices,ptr_to_data);
        CTF::Sparse_mul(&M,&M_reciprocal1);
        for (int i=0; i<num_nnz_elems; i++) ptr_to_data_t[i] = 1. + ptr_to_data[i];
        Hessian->write(num_nnz_elems,ptr_to_indices,ptr_to_data_t);
        CTF::Sparse_mul(Hessian,&M_reciprocal2);

        std::vector<CTF::Tensor<>*> lst_mat;
        std::vector<int> fm_mode_types(2,NS);
        std::vector<int> fm_mode_lengths(2,0);
        for (int j=0; j<A.size(); j++){
          if (j != num) lst_mat.push_back(A[j]);
          else{
            fm_mode_lengths[0] = A[num]->lens[0]; fm_mode_lengths[1] = A[num]->lens[1];
            lst_mat.push_back(new CTF::Tensor<>(2,&fm_mode_lengths[0],&fm_mode_types[0],*dw));
          }
        }

        CTF::MTTKRP(&M,&lst_mat[0],num,true);
        // Notice that grad should be negative, but it is not!
        //         This is taken into account when we subtract the step from the FM in 'step(..)'
        // Notice: no factors of 2. These are divided away automatically, as both the loss, regularization terms, etc. have them.
        delete[] ptr_to_data;
        delete[] ptr_to_data_t;
        delete[] ptr_to_indices;
        delete[] ptr_to_indices2;
        lst_mat[num]->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,false);
        A[num]->get_local_data(&num_nnz_elems2,&ptr_to_indices2,&ptr_to_data2,false);
        assert(num_nnz_elems==num_nnz_elems2);
        for (int i=0; i<num_nnz_elems; i++){
          ptr_to_data[i] = ptr_to_data[i] + reg*ptr_to_data2[i] - mu/2./ptr_to_data2[i];
        }
        grad->write(num_nnz_elems,ptr_to_indices,ptr_to_data);
        delete[] ptr_to_data;
        delete[] ptr_to_data2;
        delete[] ptr_to_indices;
        delete[] ptr_to_indices2;
        delete lst_mat[num];
    }

    static int step(CTF::World* dw, CTF::Tensor<>* T, CTF::Tensor<>* O, std::vector<CTF::Tensor<>*>& A, double regu, double barrier_start, double barrier_stop, double barrier_reduction_factor, double factor_matrix_convergence_tolerance, double max_newton_iter){
        int64_t num_nnz_elems,num_nnz_elems2;
        int64_t* ptr_to_indices,*ptr_to_indices2;
        double* ptr_to_data,*ptr_to_data2;
        std::vector<int> fm_mode_types(2,NS);
        std::vector<int> fm_mode_lengths(2,0);
        int newton_count = 0;

        // Sweep over each factor matrix.
        for (int i=0; i<A.size(); i++){
            std::vector<CTF::Tensor<>*> lst_mat;
            // Extract all factor matrices for this optimization (i)
            for (int j=0; j<A.size(); j++){
                if (i != j) lst_mat.push_back(A[j]);
                else{
                  fm_mode_lengths[0] = A[i]->lens[0]; fm_mode_lengths[1] = A[i]->lens[1];
                  lst_mat.push_back(new CTF::Tensor<>(2,&fm_mode_lengths[0],&fm_mode_types[0],*dw));
                }
            }
            // Minimize convex objective -> Newton's method
            // Reset barrier coefficient to starting value
            double mu = barrier_start;
            // Optimize factor matrix i by solving each row's nonlinear loss via multiple steps of Newtons method.
            while (mu >= barrier_stop){
                int t=0;
                double prev_step_nrm = 10000000.;
                while (t<max_newton_iter){
                    t += 1;
                    fm_mode_lengths[0]=A[i]->lens[0]; fm_mode_lengths[1]=A[i]->lens[1];
                    CTF::Tensor<> g(2,&fm_mode_lengths[0],&fm_mode_types[0],*dw);
                    CTF::Tensor<> Hessian(O);
                    MLogQ2::Get_RHS(dw,T,O,A,i,regu,mu,&g,&Hessian);
                    CTF::Solve_Factor(&Hessian,&lst_mat[0],&g,i,true,regu,regu,mu);
                    double step_nrm = lst_mat[i]->norm2() / A[i]->norm2();
                    prev_step_nrm = step_nrm;
                    // Verify that following update of factor matrix, every element is positive.
                    lst_mat[i]->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,false);
                    A[i]->get_local_data(&num_nnz_elems2,&ptr_to_indices2,&ptr_to_data2,false);
                    assert(num_nnz_elems == num_nnz_elems2);
                    for (int j=0; j<num_nnz_elems; j++){
                      ptr_to_data2[j] = ptr_to_data2[j] - ptr_to_data[j]; if (ptr_to_data2[j] <= 0) ptr_to_data2[j]=1e-6;
                    }
                    A[i]->write(num_nnz_elems2,ptr_to_indices2,ptr_to_data2);
                    lst_mat[i]->write(num_nnz_elems2,ptr_to_indices2,ptr_to_data2);
                    if (step_nrm <= factor_matrix_convergence_tolerance) break;
                    delete[] ptr_to_data;
                    delete[] ptr_to_data2;
                    delete[] ptr_to_indices;
                    delete[] ptr_to_indices2;
                }
                mu /= barrier_reduction_factor;
                newton_count += t;
            }
            delete lst_mat[i];
        }
        return newton_count;
    }
};

/*
struct MSE_strictly_increasing{
    //Current implementation is using \lambda  = e^m and replacing it in the function to get: e^m - xm
    static void Get_RHS(CTF::World* dw, CTF::Tensor<>* T, CTF::Tensor<>* O, std::vector<CTF::Tensor<>*>& A, int num, double reg, double mu, CTF::Tensor<>* grad, CTF::Tensor<>* Hessian){
        int64_t num_nnz_elems,num_nnz_elems2;
        int64_t* ptr_to_indices,*ptr_to_indices2;
        double* ptr_to_data,*ptr_to_data2;
        CTF::Tensor<> M(O);
        std::vector<int> mode_list(A.size());
        for (int j=0; j<A.size(); j++) mode_list[j]=j;
        CTF::TTTP(&M,A.size(),&mode_list[0],&A[0],true);
// I don't think I need this       Hessian = M.copy();
        auto M_reciprocal1 = M;
        auto M_reciprocal2 = M;

        M.get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,true);
        std::vector<double> new_data(num_nnz_elems); for (int i=0; i<num_nnz_elems; i++) new_data[i] = -1./ptr_to_data[i];
        M_reciprocal1.write(num_nnz_elems,ptr_to_indices,&new_data[0]);
        std::vector<double> new_data2(num_nnz_elems); for (int i=0; i<num_nnz_elems; i++) new_data2[i] = 1./(ptr_to_data[i]*ptr_to_data[i]);
        M_reciprocal2.write(num_nnz_elems,ptr_to_indices,&new_data2[0]);
        // Confirmed that swapping the negatives between new_data and new_data2 fails.

        double* ptr_to_data_t;
        T->get_local_data(&num_nnz_elems2,&ptr_to_indices2,&ptr_to_data_t,true);
        assert(num_nnz_elems2 == num_nnz_elems);
        for (int i=0; i<num_nnz_elems; i++) ptr_to_data[i] = log(ptr_to_data_t[i] / ptr_to_data[i]);
        M.write(num_nnz_elems,ptr_to_indices,ptr_to_data);
        CTF::Sparse_mul(&M,&M_reciprocal1);
        for (int i=0; i<num_nnz_elems; i++) ptr_to_data_t[i] = 1. + ptr_to_data[i];
        Hessian->write(num_nnz_elems,ptr_to_indices,ptr_to_data_t);
        CTF::Sparse_mul(Hessian,&M_reciprocal2);

        std::vector<CTF::Tensor<>*> lst_mat;
        std::vector<int> fm_mode_types(2,NS);
        std::vector<int> fm_mode_lengths(2,0);
        for (int j=0; j<A.size(); j++){
          if (j != num) lst_mat.push_back(A[j]);
          else{
            fm_mode_lengths[0] = A[num]->lens[0]; fm_mode_lengths[1] = A[num]->lens[1];
            lst_mat.push_back(new CTF::Tensor<>(2,&fm_mode_lengths[0],&fm_mode_types[0],*dw));
          }
        }

        CTF::MTTKRP(&M,&lst_mat[0],num,true);
        // Notice that grad should be negative, but it is not!
        //         This is taken into account when we subtract the step from the FM in 'step(..)'
        // Notice: no factors of 2. These are divided away automatically, as both the loss, regularization terms, etc. have them.
        delete[] ptr_to_data;
        delete[] ptr_to_data_t;
        delete[] ptr_to_indices;
        delete[] ptr_to_indices2;
        lst_mat[num]->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,false);
        A[num]->get_local_data(&num_nnz_elems2,&ptr_to_indices2,&ptr_to_data2,false);
        assert(num_nnz_elems==num_nnz_elems2);
        for (int i=0; i<num_nnz_elems; i++){
          ptr_to_data[i] = ptr_to_data[i] + reg*ptr_to_data2[i] - mu/2./ptr_to_data2[i];
        }
        grad->write(num_nnz_elems,ptr_to_indices,ptr_to_data);
        delete[] ptr_to_data;
        delete[] ptr_to_data2;
        delete[] ptr_to_indices;
        delete[] ptr_to_indices2;
        delete lst_mat[num];
    }

    static int step(CTF::World* dw, CTF::Tensor<>* T, CTF::Tensor<>* O, std::vector<CTF::Tensor<>*>& A, double regu, double barrier_start, double barrier_stop, double barrier_reduction_factor, double factor_matrix_convergence_tolerance, double max_newton_iter){
        int64_t num_nnz_elems,num_nnz_elems2;
        int64_t* ptr_to_indices,*ptr_to_indices2;
        double* ptr_to_data,*ptr_to_data2;
        std::vector<int> fm_mode_types(2,NS);
        std::vector<int> fm_mode_lengths(2,0);
        int newton_count = 0;
        // Sweep over each factor matrix.
        for (int i=0; i<A.size(); i++){
            std::vector<CTF::Tensor<>*> lst_mat;
            // Extract all factor matrices for this optimization (i)
            for (int j=0; j<A.size(); j++){
//              lst_mat.push_back(A[j]);
                if (i != j) lst_mat.push_back(A[j]);
                else{
                  fm_mode_lengths[0] = A[i]->lens[0]; fm_mode_lengths[1] = A[i]->lens[1];
                  lst_mat.push_back(new CTF::Tensor<>(2,&fm_mode_lengths[0],&fm_mode_types[0],*dw));
                }
            }
            // Minimize convex objective -> Newton's method
            // Reset barrier coefficient to starting value
            double mu = barrier_start;
            // Optimize factor matrix i by solving each row's nonlinear loss via multiple steps of Newtons method.
            while (mu >= barrier_stop){
                int t=0;
                double prev_step_nrm = 10000000.;
                while (t<max_newton_iter){
                    t += 1;
                    fm_mode_lengths[0]=A[i]->lens[0]; fm_mode_lengths[1]=A[i]->lens[1];
                    CTF::Tensor<> g(2,&fm_mode_lengths[0],&fm_mode_types[0],*dw);
                    CTF::Tensor<> Hessian(O);
                    MLogQ2::Get_RHS(dw,T,O,A,i,regu,mu,&g,&Hessian);
                    CTF::Solve_Factor(&Hessian,&lst_mat[0],&g,i,true,regu,regu,mu);
                    double step_nrm = lst_mat[i]->norm2() / A[i]->norm2();
//                    std::cout << mu << " " << t << " " << step_nrm << "\n";
                    prev_step_nrm = step_nrm;
                    // Verify that following update of factor matrix, every element is positive.
                    lst_mat[i]->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,false);
                    A[i]->get_local_data(&num_nnz_elems2,&ptr_to_indices2,&ptr_to_data2,false);
                    assert(num_nnz_elems == num_nnz_elems2);
                    for (int j=0; j<num_nnz_elems; j++) { ptr_to_data2[j] = ptr_to_data2[j] - ptr_to_data[j]; if (ptr_to_data2[j] <= 0) ptr_to_data2[j]=1e-6;}
                    A[i]->write(num_nnz_elems2,ptr_to_indices2,ptr_to_data2);
                    lst_mat[i]->write(num_nnz_elems2,ptr_to_indices2,ptr_to_data2);
                    if (step_nrm <= factor_matrix_convergence_tolerance) break;
                    delete[] ptr_to_data;
                    delete[] ptr_to_data2;
                    delete[] ptr_to_indices;
                    delete[] ptr_to_indices2;
                }
                mu /= barrier_reduction_factor;
                //print("Newton iteration %d: step_nrm - "%(t), step_nrm)
                //A[i] -= delta 
                newton_count += t;
            }
            delete lst_mat[i];
        }
        return newton_count;
    }
};
*/

struct MSE{
    static void Get_RHS(CTF::World* dw, CTF::Tensor<>* T, CTF::Tensor<>* O, std::vector<CTF::Tensor<>*>& A, int num, double reg, CTF::Tensor<>* grad){
  // Note: reg has been adjusted via multiplication by nnz
  std::vector<CTF::Tensor<>*> lst_mat;
  std::vector<int> fm_mode_types(2,NS);
  std::vector<int> fm_mode_lengths(2,0);
  for (int j=0; j<A.size(); j++){
    if (j != num) lst_mat.push_back(A[j]);
    else{
      fm_mode_lengths[0] = A[num]->lens[0]; fm_mode_lengths[1] = A[num]->lens[1];
      lst_mat.push_back(new CTF::Tensor<>(2,&fm_mode_lengths[0],&fm_mode_types[0],*dw));
    }
  }

  // Assume *lst_mat[num] is filled with zeros from constructor
  CTF::MTTKRP(T,&lst_mat[0],num,true);
  // Notice that grad should be negative, but it is not!
  //         This is taken into account when we subtract the step from the FM in 'step(..)'
  // Notice: no factors of 2. These are divided away automatically, as both the loss, regularization terms, etc. have them.

  int64_t num_nnz_elems,num_nnz_elems2;
  int64_t* ptr_to_indices,*ptr_to_indices2;
  double* ptr_to_data,*ptr_to_data2;
  lst_mat[num]->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,false);
  A[num]->get_local_data(&num_nnz_elems2,&ptr_to_indices2,&ptr_to_data2,false);
  assert(num_nnz_elems==num_nnz_elems2);
  for (int i=0; i<num_nnz_elems; i++){
    ptr_to_data[i] = ptr_to_data[i] - reg*ptr_to_data2[i];
  }
  grad->write(num_nnz_elems,ptr_to_indices,ptr_to_data);
  assert(ptr_to_data != ptr_to_data2);
  delete[] ptr_to_data;
  delete[] ptr_to_data2;
  delete[] ptr_to_indices;
  delete[] ptr_to_indices2;
  delete lst_mat[num];
 }

    static void step(CTF::World* dw, CTF::Tensor<>* T, CTF::Tensor<>* O, std::vector<CTF::Tensor<>*>& A, double reg, int64_t nnz){
        int64_t num_nnz_elems;
        int64_t* ptr_to_indices;
        double* ptr_to_data;
        std::vector<int> fm_mode_types(2,NS);
        std::vector<int> fm_mode_lengths(2,0);
        // Sweep over each factor matrix, stored as an entry in list 'A'
        // This outer loop determines which factor matrix we are optimizing over.
        for (int i=0; i<A.size(); i++){
            std::vector<CTF::Tensor<>*> lst_mat;
            // Extract all factor matrices for this optimization (i)
            for (int j=0; j<A.size(); j++){
                if (i != j) lst_mat.push_back(A[j]);
                else{
                  fm_mode_lengths[0] = A[i]->lens[0]; fm_mode_lengths[1] = A[i]->lens[1];
                  lst_mat.push_back(new CTF::Tensor<>(2,&fm_mode_lengths[0],&fm_mode_types[0],*dw));
                }
            }
            // Extract the rhs from a MTTKRP of the the sparse tensor T and all but the i'th factor matrix
            // MTTKRP - Matricized Tensor Times Khatri-Rao Product
            // The Tensor is T, and the Khatri-Rao Product is among all but the i'th factor matrix.
            fm_mode_lengths[0]=A[i]->lens[0]; fm_mode_lengths[1]=A[i]->lens[1];
            CTF::Tensor<> g(2,&fm_mode_lengths[0],&fm_mode_types[0],*dw);
            MSE::Get_RHS(dw,T,O,A,i,reg*nnz,&g);
            CTF::Solve_Factor(O,&lst_mat[0],&g,i,true,reg*nnz,0,0);
            lst_mat[i]->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,false);
            A[i]->write(num_nnz_elems,ptr_to_indices,ptr_to_data);
            delete[] ptr_to_data;
            delete[] ptr_to_indices;
            delete lst_mat[i];
        }
        // Updated factor matrix filled in via pointer, so return nothing
    }
};

double cpd_als(CTF::World* dw, CTF::Tensor<>* T_in, CTF::Tensor<>* O, std::vector<CTF::Tensor<>*> X, double reg, double model_convergence_tolerance, int max_nsweeps, std::string loss_function_metric="MSE"){
    assert(loss_function_metric == "MSE");

    // X - model parameters, framed as a guess
    // O - sparsity pattern encoded as a sparse matrix
    // T_in - sparse tensor of data
    // Assumption is that the error is MSE with optional regularization
    int64_t nnz;
    int64_t num_nnz_elems,num_nnz_elems2;
    int64_t* ptr_to_indices,*ptr_to_indices2;
    double* ptr_to_data,*ptr_to_data2;
    O->get_local_data(&nnz,&ptr_to_indices,&ptr_to_data,true);
    delete[] ptr_to_data;
    delete[] ptr_to_indices;
    double err=100000000.;
    int n_newton_iterations=0;
    for (int i=0; i<max_nsweeps; i++){
      if (i>0){
        // Update model parameters X, one step at a time
        MSE::step(dw,T_in,O,X,reg,nnz);
        normalize(X);
      }
        // Tensor-times-tensor product with sparsity pattern and the factor matrices
        // The point of this product is to extract the relevant approximate entries
        //    from the contracted factor matrices. If not for the TTTP with the sparsity
        //    matrix, this operation would explode memory footprint.
        CTF::Tensor<> M(O);
        std::vector<int> mode_list(X.size());
        for (int j=0; j<X.size(); j++) mode_list[j]=j;
        CTF::TTTP(&M,X.size(),&mode_list[0],&X[0],true);
        // M has same sparsity pattern as X, which has same sparsity pattern as T_in
        // Now, add M with T_in
        CTF::Sparse_add(&M,T_in,1,-1);
        err = M.norm2()/sqrt(nnz);
        err *= err;
/*
        double reg_loss = 0;
        for (int j=0; j<X.size(); j++){
            [inds,data] = X[j]->read_local_nnz();
            reg_loss += la.norm(data,2)**2;
        }
        reg_loss *= reg;
*/
        if (err < model_convergence_tolerance) break;
    }
    return err;
}

/*
double cpd_als_strictly_increasing(CTF::World* dw, CTF::Tensor<>* T_in, CTF::Tensor<>* O, std::vector<CTF::Tensor<>*> X, double reg, double model_convergence_tolerance, int max_nsweeps, double factor_matrix_convergence_tolerance, int max_newton_iter, double barrier_start=1e1, double barrier_stop=1e-11, double barrier_reduction_factor=8, std::string loss_function_metric="MSE_Strictly_Increasing"){
    assert(loss_function_metric == "MSE_Strictly_Increasing");
    int64_t nnz,num_nnz_elems;
    int64_t* ptr_to_indices;
    double* ptr_to_data;
    O->get_local_data(&nnz,&ptr_to_indices,&ptr_to_data,true);
    delete[] ptr_to_data;
    delete[] ptr_to_indices;
    reg *= nnz;
    barrier_start *= nnz;
    barrier_stop *= nnz;
    double err=100000000.;
    double err_prev = 100000000.;
    int n_newton_iterations=0;
    for (int i=0; i<max_nsweeps; i++){
        CTF::Tensor<> M(O);
        std::vector<int> mode_list(X.size());
        for (int j=0; j<X.size(); j++) mode_list[j]=j;
        CTF::TTTP(&M,X.size(),&mode_list[0],&X[0],true);
        // M has same sparsity pattern as X, which has same sparsity pattern as T_in
        // Now, add M with T_in
        CTF::Sparse_add(&M,T_in,1,-1);
        err = M.norm2()/sqrt(nnz);
        err *= err;


        //std::cout << "MLogQ2 err - " << err << " " << nnz << std::endl;
        //print("(Loss,Regularization component of objective,Objective) at AMN sweep %d: %f,%f,%f)"%(i,err,reg_loss,err+reg_loss))
        //std::cout << i << " " << err << " " << reg_loss << " " << err+reg_loss << std::endl;
        if (i>0 && abs(err)<model_convergence_tolerance) break;
        err_prev = err;
        int _n_newton_iterations = MSE_strictly_increasing::step(dw,T_in,O,X,reg,barrier_start,barrier_stop,barrier_reduction_factor,factor_matrix_convergence_tolerance, max_newton_iter);
        normalize(X);
        n_newton_iterations += _n_newton_iterations;
    }
    //print("Break with %d AMN sweeps, %d total Newton iterations"%(i,n_newton_iterations))
    return err;//,i,n_newton_iterations)
}
*/

double cpd_amn(CTF::World* dw, CTF::Tensor<>* T_in, CTF::Tensor<>* O, std::vector<CTF::Tensor<>*> X, double reg, double model_convergence_tolerance, int max_nsweeps, double factor_matrix_convergence_tolerance, int max_newton_iter, double barrier_start=1e1, double barrier_stop=1e-11, double barrier_reduction_factor=8, std::string loss_function_metric="MLogQ2"){
    assert(loss_function_metric == "MLogQ2");
    int64_t nnz,num_nnz_elems;
    int64_t* ptr_to_indices;
    double* ptr_to_data;
    O->get_local_data(&nnz,&ptr_to_indices,&ptr_to_data,true);
    delete[] ptr_to_data;
    delete[] ptr_to_indices;
    reg *= nnz;
    barrier_start *= nnz;
    barrier_stop *= nnz;
    double err=100000000.;
    double err_prev = 100000000.;
/*
    X_prev = []
    for (int i=0; i<X.size(); i++){
        X_prev.append(X[i].copy());
    }
    double err=err_prev;
*/
    int n_newton_iterations=0;
    auto TT = *T_in;
    CTF::Sparse_log(&TT);
    for (int i=0; i<max_nsweeps; i++){
      if (i>0){
        int _n_newton_iterations = MLogQ2::step(dw,T_in,O,X,reg,barrier_start,barrier_stop,barrier_reduction_factor,factor_matrix_convergence_tolerance, max_newton_iter);
        normalize(X);
        n_newton_iterations += _n_newton_iterations;
      }
        CTF::Tensor<> M(O);
        std::vector<int> mode_list(X.size());
        for (int j=0; j<X.size(); j++) mode_list[j]=j;
        CTF::TTTP(&M,X.size(),&mode_list[0],&X[0],true);
        auto P = M;
        CTF::Sparse_log(&P);
        CTF::Sparse_add(&P,&TT,-1.,1.);
        err = P.norm2(); err*=err; err/=nnz;
/*
        reg_loss = 0;
        for (int j=0 j<X.size(); j++){
            [inds,data] = X[j].read_local_nnz();
            reg_loss += la.norm(data,2)**2;
        }
        reg_loss *= (reg/nnz);
*/
/*
        if (abs(err) > 10*abs(err_prev)){
            err = err_prev;
            for (int j=0; j<X.size(); j++){
                X[j] = X_prev[j].copy();
            }
            break;
        }
*/
        if (i>0 && abs(err)<model_convergence_tolerance) break;
        err_prev = err;
/*
        for (int j=0; j<X.size(); j++){
            X_prev[j] = X[j].copy();
        }
*/
    }
    return err;
}

double get_midpoint_of_two_nodes(int idx, int num_nodes, double* _nodes, MODEL_PARAM_SPACING node_spacing_type, int ppp){
  // NOTE: 'idx' refers to the interval/partition, which reflects a single particular tensor element
  // NOTE: if node_spacing_type==SINGLE, then the midpoint of two nodes doesn't reflect the execution time of an interval.
  // NOTE: if (UNIFORM,AUTOMATIC,GEOMETRIC), 'idx' is assumed to be the leading coordinate of a cell. Therefore, 'idx'+1 is always valid
  if (node_spacing_type != SINGLE) assert(idx < num_nodes-1);
  if (node_spacing_type == GEOMETRIC || (node_spacing_type == AUTOMATIC && _nodes[idx]>1)){
      assert(_nodes[idx+1] > _nodes[idx]);
      double scale = _nodes[idx+1]*1./_nodes[idx];
      assert(scale>0);
      double transf_scale = log(scale);
      return pow(scale,(log(_nodes[idx])/transf_scale + ((log(_nodes[idx+1])/transf_scale)-(log(_nodes[idx])/transf_scale))/2.));
  } else if (node_spacing_type == UNIFORM || node_spacing_type == AUTOMATIC){
      return _nodes[idx]*1. + (_nodes[idx+1]-_nodes[idx])/2.;
  } else if (node_spacing_type == SINGLE){
      // Corner case
      if (idx==(num_nodes-1)) idx--;
      return _nodes[idx]*1. + (_nodes[idx+1]-_nodes[idx])/2.;
  } else assert(0);
}

void partition_space(int max_num_obs_per_cell, int start_idx, int end_idx, std::vector<double>& features, std::vector<double>& nodes, double max_spacing_factor){
  // Invariant: don't add the first knot in a range, and the end-points (as indices into features_copy) are always valid, and duplicate values are not present.
  assert(nodes.size()>0);
  assert(start_idx >= 0);
  assert(end_idx < features.size());
  // Below allows us to avoid consideration of cells which have no observations, even if the span of the cell is large.
  if (start_idx - end_idx >= 0) return;
  // NOTE: A small-enough number of observations in some cell is not enough to stop partitioning a cell defined by [features[start_idx],features[end_idx]].
  //       The span of the cell still matters, and if it is too large, then the quadrature error (a component of training error) will be too large!
  if (end_idx-start_idx <= max_num_obs_per_cell && features[end_idx]/features[start_idx]<=max_spacing_factor){
    if (features[end_idx] > nodes[nodes.size()-1]){
      if (nodes[nodes.size()-1] < features[start_idx]){
        nodes.push_back(features[start_idx]);
      }
      //assert(nodes[nodes.size()-1] == features[start_idx]);
      nodes.push_back(features[end_idx]);
    }
    return;
  }
  double space_range[] = {features[start_idx],features[end_idx]};
  double midpoint = get_midpoint_of_two_nodes(0,2,&space_range[0],GEOMETRIC,0);
  int midpoint_idx = std::lower_bound(features.begin()+start_idx,features.begin()+end_idx+1,midpoint) - features.begin();
  // NOTE: Ideally idpoint_idx is roughly (start_idx + (end_idx-start_idx)/2), but sometimes all of the observations exist in one of the two partitions
  if (features[midpoint_idx] >= midpoint || midpoint_idx==end_idx) --midpoint_idx;// handles corner case, prevents infinite loop
  partition_space(max_num_obs_per_cell,start_idx,midpoint_idx,features,nodes,max_spacing_factor);
  partition_space(max_num_obs_per_cell,midpoint_idx+1,end_idx,features,nodes,max_spacing_factor);
}

int get_interval_index(double val, int num_nodes, double* _nodes, MODEL_PARAM_SPACING node_spacing){
  // Used for mapping configuration into grid-cell (tensor element, equivalently)
  // NOTE: Should never be called for extrapolation! Currently an assert checks for this. The more graceful behavior would be to return the min/max interval if val exists outside.
  if (num_nodes==1) return 0;//CORNER CASE
/* Don't need these asserts anymore. If val is outside of [_nodes[0],_nodes[num_nodes-1]], then return 0 or num_nodes-1, respectively
  assert(val <= _nodes[num_nodes-1]);
  assert(val >= _nodes[0]);
*/
  assert(_nodes[0] < _nodes[num_nodes-1]);
  if (val >= _nodes[num_nodes-1]){
      return node_spacing==SINGLE ? num_nodes-1 : num_nodes-2;
  }
  if (val <= _nodes[0]) return 0;
  // Binary Search
  // Loop invariant : cell index is in [start,end]
  int start=0;
  int end=std::max(num_nodes-2,start);//NOTE: This works for SINGLE spacing too, because we already checked for the last node
  //NOTE: 'start' and 'end' represent starting indices of intervals/cells/partitions.
  while (start < end){
    int mid = start + (end-start)/2;
    assert(mid+1 < num_nodes);
    if (val >= _nodes[mid] && val < _nodes[mid+1]){
      return mid;
    }
    else if (val <= _nodes[mid]){
      end = mid;//NOTE: mid-1 also works
    }
    else if (val > _nodes[mid]){
      start = mid+1;
    }
    else{
      assert(0);
    }
  }
  return start;
}

int get_node_index(double val, int num_nodes, double* _nodes, MODEL_PARAM_SPACING node_spacing_type){
  //NOTE: This function is different than the function in python version!
  if (node_spacing_type==SINGLE) assert(0);// I just want to be alerted to this, TODO: Remove later
  if (val >= _nodes[num_nodes-1]) return num_nodes-1;
  if (val <= _nodes[0]) return 0;
  // Binary Search
  // Loop invariant : cell starting index is in [start,end]
  int start=0;
  int end=std::max(num_nodes-2,start);
  while (start < end){
    int mid = start + (end-start)/2;
    assert(mid<(num_nodes-1));
    if (val >= _nodes[mid] && val < _nodes[mid+1]){
      assert(!(mid+1==num_nodes));
      if (val <= get_midpoint_of_two_nodes(mid,num_nodes,_nodes,node_spacing_type,1)) return mid;
      else return mid+1;
    }
    else if (val < _nodes[mid]) end=mid-1;
    else start = mid+1;
  }
  return start;
}

std::vector<double> generate_nodes(double _min, double _max, int num_grid_pts, MODEL_PARAM_SPACING node_spacing_type, MODEL_PARAM_TYPE param_type){
    if (param_type==NUMERICAL && node_spacing_type==SINGLE) assert(0);// Not valid
    if (param_type==NUMERICAL){
      assert(_max > _min);
      assert(num_grid_pts>1);
    }
    else if (param_type==CATEGORICAL){
      assert(_max >= _min);
      assert(num_grid_pts>0);
    }
    std::vector<double> nodes;
    nodes.reserve(num_grid_pts);
    if (node_spacing_type == UNIFORM){
      double spacing = (_max-_min)*1./(num_grid_pts-1);
      double pos = _min;
      for (int i=0; i<num_grid_pts; i++){
        nodes.push_back(pos);
        pos += spacing;
      }
    }
    else if (node_spacing_type == GEOMETRIC){
        assert(_min>0);
        nodes = generate_nodes(log(_min),log(_max),num_grid_pts,UNIFORM,param_type);
        for (int i=0; i<nodes.size(); i++){
          nodes[i] = exp(nodes[i]);
        }
    } else assert(0);
    // Enforce the boundary points
    nodes[0]=_min;
    nodes[nodes.size()-1]=_max;
    return nodes;
}

void cp_perf_model::init(
  std::vector<int>& cells_info, const std::vector<double>& mode_range_min,
  const std::vector<double>& mode_range_max, double max_spacing_factor, const std::vector<double> custom_grid_pts,
  int num_configurations, const double* features){
    this->is_valid=true;

    for (int i=0; i<this->param_types.size(); i++){
      if (this->param_types[i]==NUMERICAL) this->numerical_modes.push_back(i);
      else this->categorical_modes.push_back(i);
    }

    this->tensor_cell_nodes.clear();
    this->tensor_cell_node_offsets.clear();
    this->tensor_mode_lengths.resize(this->order);
    this->tensor_cell_node_offsets.resize(this->order);
    this->tensor_cell_node_offsets[0]=0;
    for (int i=0; i<this->order; i++){
        if (this->interval_spacing[i]==SINGLE){
          // Valid for parameter types: CATEGORICAL and NUMERICAL
          // Each distinct value is a distinct node
          // cells_info[i] not relevant, not parsed.
          std::vector<double> temp_nodes;
          std::vector<double> features_copy(num_configurations);
          for (int j=0; j<num_configurations; j++){
            features_copy[j] = std::max(features[this->order*j+i],mode_range_min[i]);
          }
          std::sort(features_copy.begin(),features_copy.end());
          features_copy.erase(std::unique(features_copy.begin(),features_copy.end()),features_copy.end());// remove duplicates
          this->tensor_mode_lengths[i] = features_copy.size();
	  for (auto it : features_copy){
            this->tensor_cell_nodes.push_back(it);
          }
          if (i<(this->order-1)) this->tensor_cell_node_offsets[i+1]=features_copy.size()+this->tensor_cell_node_offsets[i];
        }
	else if (this->interval_spacing[i] == UNIFORM || this->interval_spacing[i] == GEOMETRIC){
          assert(this->param_types[i]==NUMERICAL);
          assert(cells_info[i]>0);
	  auto temp_nodes = generate_nodes(mode_range_min[i],mode_range_max[i],cells_info[i]+1,this->interval_spacing[i],this->param_types[i]);
          this->tensor_mode_lengths[i] = temp_nodes.size() - 1;
	  for (auto it : temp_nodes){
            if (this->interval_spacing[i]==GEOMETRIC) assert(it>0);
            this->tensor_cell_nodes.push_back(it);
          }
          assert(this->tensor_mode_lengths[i]>0);
          if (i<(this->order-1)) this->tensor_cell_node_offsets[i+1]=temp_nodes.size()+this->tensor_cell_node_offsets[i];
        } else if (this->interval_spacing[i] == AUTOMATIC){
            assert(this->param_types[i]==NUMERICAL);
            assert(num_configurations > 0);
            //NOTE: cells_info[i] can be positive or negative
            int max_num_distinct_obs_per_cell = cells_info[i]<0 ? cells_info[i]*(-1) : num_configurations/cells_info[i];
            std::vector<double> temp_nodes;
            std::vector<double> features_copy(num_configurations);
            for (int j=0; j<num_configurations; j++){
              features_copy[j] = std::max(features[this->order*j+i],mode_range_min[i]);
            }
            std::sort(features_copy.begin(),features_copy.end());
            features_copy.erase(std::unique(features_copy.begin(),features_copy.end()),features_copy.end());// remove duplicates
            temp_nodes.push_back(mode_range_min[i]);
            // Invariant: don't add the first knot in a range, and the end-points (as indices into features_copy) are always valid, and duplicate values are not present.
            partition_space(max_num_distinct_obs_per_cell,0,features_copy.size()-1,features_copy,temp_nodes,max_spacing_factor);
            if (temp_nodes.size()==1 || temp_nodes[temp_nodes.size()-1] < mode_range_max[i]){
              temp_nodes.push_back(mode_range_max[i]);
            }
            assert(temp_nodes.size()>1);
            assert(temp_nodes[temp_nodes.size()-1] == mode_range_max[i]);
            // Below handles a corner case in which all parameter values are the same.
            if (temp_nodes.size()==2 && temp_nodes[1]==temp_nodes[0]){
              assert(0);
              ++temp_nodes[1];
            }
            assert(temp_nodes[0]==mode_range_min[i]);
            //assert(temp_nodes.size()>1);
 	    for (int j=0; j<temp_nodes.size(); j++){ if (j>0) { assert(temp_nodes[j]>temp_nodes[j-1]); } this->tensor_cell_nodes.push_back(temp_nodes[j]); }
            this->tensor_mode_lengths[i] = temp_nodes.size()-1;
            if (this->param_types[i]==NUMERICAL) assert(this->tensor_mode_lengths[i]>0);
            if (i<(this->order-1)) this->tensor_cell_node_offsets[i+1]=temp_nodes.size()+this->tensor_cell_node_offsets[i];
 	} else{
	    assert(0);//CUSTOM interval spacing not tested yet.
            assert(cells_info[i]>0);
	    //auto temp_nodes = generate_nodes(mode_range_min[i],mode_range_max[i],cells_info[i],this->interval_spacing[i],std::vector<double>(this->custom_grid_pts.begin()+start_grid_idx,this->custom_grid_pts.begin()+start_grid_idx+cells_info[i]]));
	    auto temp_nodes = generate_nodes(mode_range_min[i],mode_range_max[i],cells_info[i],this->interval_spacing[i],this->param_types[i]);
            this->tensor_mode_lengths[i] = temp_nodes.size();
	    for (auto it : temp_nodes) this->tensor_cell_nodes.push_back(it);
            if (i<(this->order-1)) this->tensor_cell_node_offsets[i+1]=temp_nodes.size()+this->tensor_cell_node_offsets[i];
	}
        assert(this->tensor_mode_lengths[i]>0);
    }
/*
    // Debug Info
    std::cout << "Min/Max mode range: " << std::endl;
    for (int i=0; i<this->order; i++){
      std::cout << mode_range_min[i] << " " << mode_range_max[i] << std::endl;
    }
    std::cout << "cells per dim:\n";
    for (int i=0; i<this->order; i++){
      std::cout << cells_info[i] << " ";
    }
    std::cout << std::endl;
    int temp_idx = 0;
    for (int i=0; i<this->order; i++){
      for (int j=0; j<this->tensor_mode_lengths[i]+(this->param_types[i]==NUMERICAL ? 1 : 0); j++){
        std::cout << this->tensor_cell_nodes[temp_idx++] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << this->cp_rank[0] << " " << this->cp_rank[1] << " " << this->response_transform << std::endl;
    for (int i=0; i<this->order; i++) std::cout << this->tensor_mode_lengths[i] << " ";
    std::cout << std::endl;
*/
    this->Projected_Omegas.clear();
    this->Projected_Omegas.resize(this->order);

}

cp_perf_model::cp_perf_model(int order, std::vector<MODEL_PARAM_TYPE> param_types,
  std::vector<MODEL_PARAM_SPACING> interval_spacing, std::vector<int> cp_rank, int response_transform, bool save_dataset){
  // Inspect interval_spacing to make sure none are AUTOMATIC, because we have not implemented that yet.
  assert(param_types.size() == order);
  assert(interval_spacing.size() == order);
  for (auto it : interval_spacing) assert(it != CUSTOM);
  for (int i=0; i<order; i++) { if (param_types[i]!=NUMERICAL) { assert(0); } }// NOT TESTED YET
  this->order = order;
  assert(this->order>0);
  this->save_dataset = save_dataset;
  this->param_types = param_types;
  this->interval_spacing = interval_spacing;
  this->response_transform = response_transform;
  this->cp_rank[0] = cp_rank[0];
  this->cp_rank[1] = cp_rank[1];
  this->is_valid=false;
  this->factor_matrix_elements_generalizer = nullptr;
  this->factor_matrix_elements_extrapolator = nullptr;
  this->factor_matrix_elements_extrapolator_models = nullptr;
  this->timer1=0;
  this->timer2=0;
  this->timer3=0;
  this->timer4=0;
  this->timer5=0;
  this->timer6=0;
  this->timer7=0;
}

cp_perf_model::cp_perf_model(int order, const std::vector<int>& cells_info, std::vector<MODEL_PARAM_TYPE> param_types,
  std::vector<MODEL_PARAM_SPACING> interval_spacing, const std::vector<double>& mode_range_min,
  const std::vector<double>& mode_range_max, std::vector<int> cp_rank,
  int response_transform, double max_spacing_factor, std::vector<double> custom_grid_pts,
  bool save_dataset) : cp_perf_model(order,param_types,interval_spacing,cp_rank,response_transform,save_dataset){
    assert(0);//TODO
    for (int i=0; i<order; i++){
      assert(mode_range_min[i] <= mode_range_max[i]);
    }
    //TODO: Add more asserts on inputs.
    std::vector<int> local_cell_info = cells_info;
    this->init(local_cell_info,mode_range_min,mode_range_max,max_spacing_factor,custom_grid_pts);
}

cp_perf_model::~cp_perf_model(){
  if (this->factor_matrix_elements_generalizer != nullptr) delete[] this->factor_matrix_elements_generalizer;
  if (this->factor_matrix_elements_extrapolator != nullptr) delete[] this->factor_matrix_elements_extrapolator;
  if (this->factor_matrix_elements_extrapolator_models != nullptr) delete[] this->factor_matrix_elements_extrapolator_models;
  this->factor_matrix_elements_generalizer=nullptr;
  this->factor_matrix_elements_extrapolator=nullptr;
  this->factor_matrix_elements_extrapolator_models=nullptr;
}

cp_perf_model::cp_perf_model(std::string& file_path){
      assert(0);
      std::ifstream model_file_ptr;
      model_file_ptr.open(file_path,std::ios_base::out);
      if(model_file_ptr.fail()){ this->is_valid = false; return; }
      this->is_valid=true;
      std::string temp;
      getline(model_file_ptr,temp,',');
      this->order = std::stoi(temp);
      this->tensor_mode_lengths.resize(this->order);
      getline(model_file_ptr,temp,'\n');
      this->cp_rank[0] = std::stoi(temp);
      //NOTE: The number of knots (equivalently, grid-pts) per mode is NOT equivalent to the number of tensor elements per mode. If there are N knots per mode, then the number of tensor elements, or sub-intervals equivalently, is N-1.
      int num_knots = 0;
      for (int i=0; i<this->order-1; i++){
	getline(model_file_ptr,temp,',');
	this->tensor_mode_lengths[i]=std::stoi(temp)-1;
	num_knots += (1+this->tensor_mode_lengths[i]);
      }
      getline(model_file_ptr,temp,'\n');
      this->tensor_mode_lengths[this->order-1]=std::stoi(temp)-1;
      num_knots += (1+this->tensor_mode_lengths[this->order-1]);
      this->tensor_cell_nodes.resize(num_knots);
      //TODO: I am skeptical about the code below, use of num_knots should be instead num_cells, which is then dependent on the param_type, right?
      //this->factor_matrix_elements_generalizer = new double[(num_knots-this->order)*cp_rank[0]];
      //this->factor_matrix_elements_extrapolator = new double[...];
      int node_counter=0;
      int factor_matrix_element_counter=0;
      for (int i=0; i<this->order; i++){
	getline(model_file_ptr,temp,',');
	this->tensor_cell_nodes[node_counter++] = std::stod(temp);
	for (int j=0; j<this->cp_rank[0]-1; j++){
	  getline(model_file_ptr,temp,',');
	  this->factor_matrix_elements_generalizer[factor_matrix_element_counter++] = std::stod(temp);
	}
	getline(model_file_ptr,temp,'\n');
	this->factor_matrix_elements_generalizer[factor_matrix_element_counter++] = std::stod(temp);
      }
      //TODO: Read in extrapolation model parameters to factor_matrix_elements_extrapolator, but this requires writer to write it first.
      this->save_dataset=false;
      //TODO: All below should be read from file
/*
      this->response_transform=1;
      std::array<int,2> cp_rank;
      std::vector<int> tensor_mode_lengths;
      std::vector<int> interval_spacing;
      std::vector<int> param_types;
      std::vector<double> tensor_cell_nodes;
      std::vector<int> numerical_modes;
      std::vector<int> categorical_modes;
*/
      //NOTE: Below: not sure whether to store all configurations
      this->yi.clear();
      this->Xi.clear();
      this->Projected_Omegas.clear();
      this->Projected_Omegas.resize(this->order);
}

int cp_perf_model::get_cp_rank(){
  return this->cp_rank[0];
}

double cp_perf_model::predict(const double* configuration, const double* _factor_matrix_elements_generalizer_, const double* _factor_matrix_elements_extrapolator_){
      // NOTE: Assumes that len of configuration is equal to this->order
      // NOTE: The below is still valid, we just need to make sure that the training always works, even if one of the modes has a range of 0. This is a separate robustness check.
      const double* fmg = _factor_matrix_elements_generalizer_==nullptr ? this->factor_matrix_elements_generalizer : _factor_matrix_elements_generalizer_;
      const double* fme = _factor_matrix_elements_extrapolator_==nullptr ? this->factor_matrix_elements_extrapolator : _factor_matrix_elements_extrapolator_;
      const double* fmesvd = this->factor_matrix_elements_extrapolator_models;
      assert(fmg!=nullptr);
      assert(fme!=nullptr);


      double max_interpolation_factor_tol = INTERPOLATION_FACTOR_TOL;
      char* env_var_ptr = std::getenv("CPPM_INTERPOLATION_FACTOR_TOL");
      if (env_var_ptr != NULL) max_interpolation_factor_tol = atof(env_var_ptr);

      // Debug
      int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
      bool do_i_want_to_print_out = false;//world_rank==0 && this->tensor_mode_lengths[0]>10 && this->tensor_mode_lengths[1]>10;

      // Check for invalid input
      for (int i=0; i<this->order; i++){
        if (configuration[i]<=0 && this->param_types[i]==NUMERICAL) return MIN_POS_RUNTIME;// default "zero-value"
      }

      std::vector<int> node(this->order,-1);
      std::vector<double> midpoints;
      std::vector<int> intervals;
      std::vector<int> modes_to_interpolate;
      std::vector<double> local_interp_map(this->order,0);
      std::vector<int> decisions(this->order,0);
      bool is_extrapolation = false;
      for (int j=0; j<this->order; j++){
          // Get the closest node (note that node!=midpoint). Two nodes define the bounding box of a grid-cell. Each grid-cell has a mid-point.
          if (this->param_types[j]==CATEGORICAL){
            assert(0);
            // Extrapolation is not possible here
            node[j] = get_interval_index(configuration[j],this->tensor_mode_lengths[j],&this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]],this->fixed_interval_spacing[j]);/*get_node_index(configuration[j],this->tensor_mode_lengths[j]+(this->param_types[j]==NUMERICAL ? 1 : 0),&this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]],this->fixed_interval_spacing[j]);*/
            continue;
          }
          if (this->tensor_mode_lengths[j]==1){
            //The reason we treat this as a specal case is because linear interpolation is not valid, as there is just one point. Linear extraplation is also invalid. 
            // Further, extrapolation is just not possible in this case, so we don't even check for it.
            node[j] = 0;
            continue;
          }
          // check if configuration[numerical_modes[j]] is outside of the parameter_nodes on either side
          double leftmost_midpoint = get_midpoint_of_two_nodes(0, this->tensor_mode_lengths[j]+(this->fixed_interval_spacing[j]==SINGLE ? 0 : 1),&this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]], this->fixed_interval_spacing[j],3);
          assert(!(this->tensor_mode_lengths[j]-(this->fixed_interval_spacing[j]==SINGLE ? 2 : 1)+1==this->tensor_mode_lengths[j]+(this->fixed_interval_spacing[j]==SINGLE ? 0 : 1)));
          double rightmost_midpoint = get_midpoint_of_two_nodes(this->tensor_mode_lengths[j]-(this->fixed_interval_spacing[j]==SINGLE ? 2 : 1), this->tensor_mode_lengths[j]+(this->fixed_interval_spacing[j]==SINGLE ? 0 : 1), &this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]], this->fixed_interval_spacing[j],4);
          int interval_idx = get_interval_index(configuration[j],this->tensor_mode_lengths[j]+(this->fixed_interval_spacing[j]==SINGLE ? 0 : 1),&this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]],this->fixed_interval_spacing[j]);
          if (do_i_want_to_print_out){
            std::cout << "leftmost midpoint: " << leftmost_midpoint << ", rightmost midpoint - " << rightmost_midpoint << std::endl;
          }
          if (configuration[j] < this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]]){
              // extrapolation necessary: outside range of bounding box on left
              decisions[j]=3;
              is_extrapolation=true;
          } else if (configuration[j] > this->tensor_cell_nodes[j==(this->order-1) ? this->tensor_cell_nodes.size()-1 : this->tensor_cell_node_offsets[j+1]-1]){
              // extrapolation necessary: outside range of bounding box on right
              decisions[j]=4;
              is_extrapolation=true;
          } else if (configuration[j] < leftmost_midpoint && this->fixed_interval_spacing[j]!=SINGLE){
              // extrapolation necessary: inside range of bounding box on left, but left of left-most midpoint
              if (do_i_want_to_print_out) std::cout << "Linear extrapolation left\n";
              decisions[j]=1;
          } else if (configuration[j] > rightmost_midpoint && this->fixed_interval_spacing[j]!=SINGLE){
              // extrapolation necessary: inside range of bounding box on right, but right of right-most midpoint
              if (do_i_want_to_print_out) std::cout << "Linear extrapolation right\n";
              decisions[j]=2;
          } else if (this->fixed_interval_spacing[j]==SINGLE && configuration[j]==this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]+interval_idx]){
              // Don't even attempt to interpolate in this case, because the result will be the same. This avoids unecessary work.
              node[j]=interval_idx;
          } else{
              // Safeguard checks: use raw factor matrix elements rather than recontructing the entire element, as we want to fix all other factor matrix values anyways!
              bool is_valid_to_interpolate = true;
              // Must check all values in corresponding FM row, not just first.
              int eleft = std::max(0,interval_idx-1);
              int emid = interval_idx;
              int eright = std::min(interval_idx+1,this->tensor_mode_lengths[j]-1);
              int fm_offset = 0;
              for (int k=0; k<j; k++) fm_offset += this->tensor_mode_lengths[k]*this->cp_rank[0];
              for (int k=0; k<this->cp_rank[0]; k++){
                int eleft_idx = fm_offset+eleft*this->cp_rank[0]+k;
                int emid_idx = fm_offset+emid*this->cp_rank[0]+k;
                int eright_idx = fm_offset+eright*this->cp_rank[0]+k;
                //NOTE: Use of fmg here is fine, even if we end up using fme
                if (fmg[eleft_idx]*fmg[emid_idx]<0) { is_valid_to_interpolate = false; break; }
                if (abs(log(abs(fmg[eleft_idx]/abs(fmg[emid_idx])))) > max_interpolation_factor_tol) { is_valid_to_interpolate = false; break; }
                if (fmg[eright_idx]*fmg[emid_idx]<0) { is_valid_to_interpolate = false; break; }
                if (abs(log(abs(fmg[eright_idx]/abs(fmg[emid_idx])))) > max_interpolation_factor_tol) { is_valid_to_interpolate = false; break; }
              }
              if (!is_valid_to_interpolate){
                node[j]=interval_idx;
              } else{
                // Feature value configuration[j] exists within mid-points, so linear interpolation of tensor elements (which represent the sample mean execution time of the midpoint) is feasible
                intervals.push_back(interval_idx);
                //NOTE: The actual midpoint isn't obvious.
                assert(intervals[intervals.size()-1]>=0);
                if (this->fixed_interval_spacing[j]!=SINGLE) assert(!(intervals[intervals.size()-1]+1==this->tensor_mode_lengths[j]+(this->fixed_interval_spacing[j]==SINGLE ? 0 : 1)));
                midpoints.push_back(get_midpoint_of_two_nodes(intervals[intervals.size()-1], this->tensor_mode_lengths[j]+(this->fixed_interval_spacing[j]==SINGLE ? 0 : 1),&this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]], this->fixed_interval_spacing[j],10));
                if (do_i_want_to_print_out) std::cout << "Interpolation, interval - " << intervals[intervals.size()-1] << ", midpoint - " << midpoints[midpoints.size()-1] << " " << this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]+intervals[intervals.size()-1]] << "," << this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]+intervals[intervals.size()-1]+1] << std::endl;
                modes_to_interpolate.push_back(j);
                local_interp_map[j] = 1;
                decisions[j]=5;
             }
           }
         }

      const double* fm = (is_extrapolation ? fme : fmg);
      int rank = (is_extrapolation ? this->cp_rank[1] : this->cp_rank[0]);

      std::vector<std::pair<int,int>> element_index_modes_list;
      // Because each tensor element is associated with the mid-point of the corresponding grid-cell (as a tensor product of mid-points of each parameter's range), we also want to get the surrounding nodes
      // calculation of the nodes to use for a (multi)linear interpolation strategy is nontrivial
      for (int j=0; j<modes_to_interpolate.size(); j++){
        // NOTE: This loop iteration exists only if there always exists a midpoint to left and right!
        element_index_modes_list.push_back(std::make_pair(-1,-1));
        if (this->fixed_interval_spacing[modes_to_interpolate[j]]==SINGLE){
	    element_index_modes_list[j].first = std::min(this->tensor_mode_lengths[modes_to_interpolate[j]]-2,intervals[j]);
	    element_index_modes_list[j].second = element_index_modes_list[j].first+1;
        } else{
	  if (configuration[modes_to_interpolate[j]] < midpoints[j]){//TODO: Watch for this "< vs <=" sign
	    element_index_modes_list[j].first = std::min(this->tensor_mode_lengths[modes_to_interpolate[j]]-2,intervals[j]-1);
	    element_index_modes_list[j].second = element_index_modes_list[j].first+1;
	  } else{
	    element_index_modes_list[j].first = std::min(intervals[j],this->tensor_mode_lengths[modes_to_interpolate[j]]-2);
	    element_index_modes_list[j].second = element_index_modes_list[j].first+1;
          }
	}
      }

        double model_val = 0.;
        // Do not consider extrapolation modes
        int num_interp_pts = (1<<modes_to_interpolate.size());
        for (int j=0; j<num_interp_pts; j++){
          int interp_id = j;
          std::vector<int> interp_id_list(modes_to_interpolate.size(),0);
          int counter = 0;
          while (interp_id>0){
	    assert(counter < interp_id_list.size());
	    interp_id_list[counter] = interp_id%2;
	    interp_id/=2;
	    counter++;
	  }
          // interp_id_list stores the particular mid-point whose execution time we are using as part of multilinear interpolation strategy
	  double coeff = 1;
	  for (int l=0; l<modes_to_interpolate.size(); l++){
	    int cell_node_idx = this->tensor_cell_node_offsets[modes_to_interpolate[l]];
	    if (interp_id_list[l] == 0){
	      //TODO: Check the correctness here
              int order_id = modes_to_interpolate[l];
	      int& temp_id = element_index_modes_list[l].first;
              assert(temp_id>=0);
              double left_point = this->fixed_interval_spacing[order_id]==SINGLE ? this->tensor_cell_nodes[this->tensor_cell_node_offsets[order_id]+temp_id] : get_midpoint_of_two_nodes(temp_id, this->tensor_mode_lengths[order_id]+1,&this->tensor_cell_nodes[this->tensor_cell_node_offsets[order_id]], this->fixed_interval_spacing[order_id],11);
              if (this->fixed_interval_spacing[order_id]!=SINGLE) assert(!(temp_id+2==this->tensor_mode_lengths[order_id]+1));
              double right_point = this->fixed_interval_spacing[order_id]==SINGLE ? this->tensor_cell_nodes[this->tensor_cell_node_offsets[order_id]+temp_id+1] : get_midpoint_of_two_nodes(temp_id+1, this->tensor_mode_lengths[order_id]+1,&this->tensor_cell_nodes[this->tensor_cell_node_offsets[order_id]], this->fixed_interval_spacing[order_id],12);
              if (do_i_want_to_print_out) std::cout << "Points(0): " << left_point << " " << right_point << std::endl;
	      coeff *= std::max(0.,(1-(abs(configuration[order_id]-left_point))/(right_point-left_point)));
              assert(coeff >= 0);
	    }
            // coeff quantifies how close the current point (configuration) is to the mid-point characterized by interp_id_list
	    if (interp_id_list[l] == 1){
	      //TODO: Check the correctness here
              int order_id = modes_to_interpolate[l];
              int& temp_id = element_index_modes_list[l].second;
              assert(temp_id>=1);
              if (!this->fixed_interval_spacing[order_id]==SINGLE) assert(!(temp_id==this->tensor_mode_lengths[order_id]+1));
              double left_point = this->fixed_interval_spacing[order_id]==SINGLE ? this->tensor_cell_nodes[this->tensor_cell_node_offsets[order_id]+temp_id-1] : get_midpoint_of_two_nodes(temp_id-1, this->tensor_mode_lengths[order_id]+1,&this->tensor_cell_nodes[this->tensor_cell_node_offsets[order_id]], this->fixed_interval_spacing[order_id],13);
              if (!this->fixed_interval_spacing[order_id]==SINGLE) assert(!(temp_id+1==this->tensor_mode_lengths[order_id]+1));
              double right_point = this->fixed_interval_spacing[order_id]==SINGLE ? this->tensor_cell_nodes[this->tensor_cell_node_offsets[order_id]+temp_id] : get_midpoint_of_two_nodes(temp_id, this->tensor_mode_lengths[order_id]+1,&this->tensor_cell_nodes[this->tensor_cell_node_offsets[order_id]], this->fixed_interval_spacing[order_id],14);
              if (do_i_want_to_print_out) std::cout << "Points(1): " << left_point << " " << right_point << std::endl;
	      coeff *= std::max(0.,(1-(abs(configuration[order_id]-right_point))/(right_point-left_point)));
              assert(coeff >= 0);
	    }
	  }
	  std::vector<double> factor_row_list;
//	  factor_row_list.reserve(rank*this->order);
	  int interp_counter = 0;
	  int factor_matrix_offset = 0;
          int fmesvd_offset=0;
          // Concatenate all of the factor matrix elements necessary to perform multilinear product
          for (int l=0; l<this->order; l++){
            if (local_interp_map[l]==1){
              int& temp_id = interp_id_list[interp_counter]==0 ? element_index_modes_list[interp_counter].first : element_index_modes_list[interp_counter].second;
              for (int ll=0; ll<rank; ll++){
                factor_row_list.push_back(fm[factor_matrix_offset + temp_id*rank+ll]);
              }
              interp_counter++;
            } else{
              if (decisions[l]==0){
                // categorical or non-numerical parameter in which interpolation/extrapolation is not relevant
                for (int ll=0; ll<rank; ll++){
                  assert(node[l]<this->tensor_mode_lengths[l]);// especially important if tensor_mode_lengths[l]==1
                  factor_row_list.push_back(fm[factor_matrix_offset + node[l]*rank+ll]);
                }
              } else if (decisions[l]==1){
                assert(this->fixed_interval_spacing[l]!=SINGLE);
                //TODO: Why do it this way? Why not the factor matri value and apply coefficient? I guess it might be the same thing, since we only care about one side? Verify this.
                double left_midpoint = get_midpoint_of_two_nodes(0, this->tensor_mode_lengths[l]+1,&this->tensor_cell_nodes[this->tensor_cell_node_offsets[l]], this->fixed_interval_spacing[l],15);
                double right_midpoint = get_midpoint_of_two_nodes(1, this->tensor_mode_lengths[l]+1,&this->tensor_cell_nodes[this->tensor_cell_node_offsets[l]], this->fixed_interval_spacing[l],16);
                for (int ll=0; ll<rank; ll++){
                  //TODO: should use log or non-log transformation of right_midpoint,left_midpoint, configuration[l] and left_midpoint? Based on fixed_interval_spacing and whether configuration[l]==0?
                  double _slope = (fm[factor_matrix_offset+rank+ll]-fm[factor_matrix_offset+ll])/(right_midpoint - left_midpoint);
                  double _num = configuration[l]-left_midpoint;
                  assert(_num <= 0);
                  factor_row_list.push_back(fm[factor_matrix_offset+ll] + _num*_slope);
                }
              } else if (decisions[l]==2){
                assert(this->fixed_interval_spacing[l]!=SINGLE);
                //TODO: Why do it this way? Why not the factor matri value and apply coefficient? I guess it might be the same thing, since we only care about one side? Verify this.
                double left_midpoint = get_midpoint_of_two_nodes(this->tensor_mode_lengths[l]-2, this->tensor_mode_lengths[l]+1, &this->tensor_cell_nodes[this->tensor_cell_node_offsets[l]], this->fixed_interval_spacing[l],19);
                double right_midpoint = get_midpoint_of_two_nodes(this->tensor_mode_lengths[l]-1, this->tensor_mode_lengths[l]+1, &this->tensor_cell_nodes[this->tensor_cell_node_offsets[l]], this->fixed_interval_spacing[l],21);
                for (int ll=0; ll<rank; ll++){
                  double _slope = (fm[factor_matrix_offset+(this->tensor_mode_lengths[l]-1)*rank+ll]-fm[factor_matrix_offset+(this->tensor_mode_lengths[l]-2)*rank+ll]);
                  double _denom = right_midpoint - left_midpoint;
                  double _num = configuration[l]-right_midpoint;
                  assert(_num >= 0);
                  factor_row_list.push_back(fm[factor_matrix_offset+(this->tensor_mode_lengths[l]-1)*rank+ll] + _num/_denom*_slope);
                }
              } else{
                  assert(rank == this->cp_rank[1]);
                  double ppp = 0;
                  double input_scale=1;
                  for (int kk=0; kk<1+this->max_spline_degree; kk++){
                    ppp += input_scale*fmesvd[fmesvd_offset+kk];
                    input_scale*=log(configuration[l]);
                  }
                  for (int lll=0; lll<rank; lll++){
                    factor_row_list.push_back(exp(ppp)*fmesvd[fmesvd_offset+1+this->max_spline_degree]*fmesvd[fmesvd_offset+2+this->max_spline_degree+lll]);
                  }
            }
          }
          factor_matrix_offset += this->tensor_mode_lengths[l]*rank;
          if (this->param_types[l]==NUMERICAL) fmesvd_offset += (2+this->max_spline_degree+rank);
        }
        double t_val = multilinear_product_packed(&factor_row_list[0],factor_row_list.size(),this->order,this->cp_rank[0]);
        if (this->response_transform==1){
          t_val = exp(t_val);
        }
        if (do_i_want_to_print_out) std::cout << "Coeff - " << coeff << std::endl;
        model_val += coeff * t_val;
      }
      if (do_i_want_to_print_out){
        std::cout << model_val << " " << decisions[0] << " " << decisions[1] << " " << configuration[0] << " " << configuration[1] << std::endl;
      }
      //assert(model_val>0);
      return std::max(MIN_POS_RUNTIME,model_val);// Only way MIN_POS_RUNTIME is used is if linear extrapolation (decisions 1 or 2) is used  and it is inaccurate.
      assert(0); return 1;
}

bool cp_perf_model::train(bool compute_fit_error, double* loss, int num_configurations, const double* configurations, const double* runtimes, MPI_Comm cm_training, MPI_Comm cm_data, std::vector<int>& cells_info, const char* loss_function, bool aggregate_obs_across_comm, double max_spacing_factor, std::vector<double> regularization, int max_spline_degree, std::vector<double> model_convergence_tolerance,
      std::vector<int> maximum_num_sweeps, double factor_matrix_convergence_tolerance,
      int maximum_num_iterations, std::vector<double> barrier_range,
      double barrier_reduction_factor, std::vector<int> projection_set_size_threshold_){

      if (this->order<=1) return false;//TODO: TEMPORARY, there is a bug with this->order==1

      int my_rank; MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);//TODO: Remove later
      int world_rank = my_rank;//TODO: delete later
      int world_size_for_training,world_size_for_data_aggregation;
      MPI_Comm_size(cm_training,&world_size_for_training);
      MPI_Comm_size(cm_data,&world_size_for_data_aggregation);
      assert(world_size_for_training==1);//TODO: Lift this constraint later
      double start_time = get_wall_time();

      const double* save_c_ptr = configurations;
      const double* save_r_ptr = runtimes;

      bool delete_data = false;
      if (aggregate_obs_across_comm){
        delete_data = aggregate_observations(num_configurations,configurations,runtimes,this->order,cm_data);
        if (delete_data){
          assert(configurations != save_c_ptr);
          assert(runtimes != save_r_ptr);
        } else{
          if (num_configurations>0){
            assert(configurations == save_c_ptr);
            assert(runtimes == save_r_ptr);
          }
        }
      }

      int min_num_obs_for_training = MIN_NUM_OBS_FOR_TRAINING;
      char* env_var_ptr = std::getenv("CPPM_MIN_NUM_OBS_FOR_TRAINING");
      if (env_var_ptr != NULL){
        min_num_obs_for_training = atoi(env_var_ptr);
        assert(min_num_obs_for_training>0);
      }
      //TODO: Average into sample mean here, rather than waiting for later, as we do not need to find a node_key yet!
      // Check how many distinct configurations there are.
      std::set<std::vector<double>> distinct_configuration_dict;
      std::vector<double> config(this->order);
      for (int i=0; i<num_configurations; i++){
        for (int j=0; j<this->order; j++){
          config[j]=configurations[i*this->order+j];
        }
        distinct_configuration_dict.insert(config);
      }
      loss[2] = distinct_configuration_dict.size();
      // No point in training if the sample size is so small
      // We only really care about distinct configurations
      if (distinct_configuration_dict.size() < min_num_obs_for_training){
        if (delete_data){
          delete[] configurations;
          delete[] runtimes;
        }
        loss[3+this->order]=-1;
        loss[4+this->order]=-1;
        return false;
      }
      distinct_configuration_dict.clear();// TODO: Later don't do this

      assert(this->order > 0);// order must be set, even if is_valid==false.
      std::vector<double> mode_range_min(this->order,INT_MAX);
      std::vector<double> mode_range_max(this->order,INT_MIN);
      for (int i=0; i<num_configurations; i++){
	for (int j=0; j<this->order; j++){
	  mode_range_min[j]=std::min(mode_range_min[j],configurations[i*this->order+j]);
	  mode_range_max[j]=std::max(mode_range_max[j],configurations[i*this->order+j]);
	}
      }
      for (int j=0; j<this->order; j++) if (mode_range_max[j] < mode_range_min[j]){
        if (delete_data){
          delete[] configurations;
          delete[] runtimes;
        }
        loss[3+this->order] = -1;
        loss[4+this->order] = -1;
        return false;
      }
      // Save member variables
      auto save_interval_spacing = this->interval_spacing;
      // Update the interval spacing justs in case it was set incorrectly (as default is geometric)
      // NOTE: The user may have specified GEOMETRIC spacing for a NUMERICAL parameter, yet the range is so small that it is clear that a log-like tranformation
      //    was already applied to the parameter values in the training set (e.g., if process count has been transformed to log(process count). In that case,
      //    the spacing should be reset to UNIFORM and a zero value should be allowed.
      for (int j=0; j<this->order; j++){
        if (this->param_types[j]==NUMERICAL && this->interval_spacing[j]==GEOMETRIC && mode_range_min[j] <= 0){
          this->interval_spacing[j]=UNIFORM;
          assert(cells_info[j]>0);
        }
      }
      for (int j=0; j<this->order; j++){
        if (this->param_types[j]==NUMERICAL && mode_range_min[j]==mode_range_max[j]){
          this->interval_spacing[j]=SINGLE;
        }
      }

      //NOTE: Ideally if we have some mode/parameter with a single distinct observation, we would enforce that its type is CATEGORICAL rather than NUMERICAL. However, this would not allow ourselves to be robust to future input configurations in which the value for that parameter is different.

      // If the max boundary value of the range is simply too small for the interval spacing of NUMERICAL parameter to be GEOMETRIC, then reset to UNIFORM.
      // TODO:  In this case, it might also be useful to reset to AUTOMATIC so as to leverage the fact that many values in the (small) range might be unobserved.
      for (int j=0; j<this->order; j++){
        if ((this->param_types[j]==NUMERICAL) && (mode_range_max[j]<NUMERICAL_PARAM_MIN_OBS_RANGE)){
          this->interval_spacing[j]=SINGLE;
          cells_info[j]=mode_range_max[j]-mode_range_min[j]+1;
        }
      }
      //NOTE: There is also a corner case in which the parameter type is NUMERICAL and the interval spacing is GEOMETRIC, yet the observed feature values are large with a very small range.
      //      We keep GEOMETRIC spacing, noting that the GEOMETRIC spacing is very similar to UNIFORM for this use case.
      std::vector<int> local_cells_info(this->order);
      for (int j=0; j<this->order; j++){
	//NOTE: Below handles a corner case
	local_cells_info[j] = (this->interval_spacing[j]==AUTOMATIC ? cells_info[j] : static_cast<int>(std::min(static_cast<double>(cells_info[j]),mode_range_max[j]-mode_range_min[j]+1)));
      }
      this->fixed_interval_spacing = this->interval_spacing;
      this->timer1 += (get_wall_time() - start_time);
      start_time = get_wall_time();
      this->init(local_cells_info,mode_range_min,mode_range_max,max_spacing_factor,{},num_configurations,configurations);

      this->timer2 += (get_wall_time() - start_time);
      for (int j=0; j<this->order; j++){
        loss[3+j] = this->tensor_mode_lengths[j];
      }
      start_time = get_wall_time();

      if (this->save_dataset){
        this->Xi.clear();
        this->yi.clear();
        for (int i=0; i<num_configurations; i++){
          this->yi.push_back(runtimes[i]);
          for (int j=0; j<this->order; j++) this->Xi.push_back(configurations[this->order*i+j]);
        }
      }
      if (projection_set_size_threshold_.size() < this->order){
        projection_set_size_threshold_.resize(this->order,1);	// very small estimate
      }
      // Use dictionaries to save the sizes of Omega_i
      assert(this->Projected_Omegas.size() == this->order);
      for (int i=0; i<this->order; i++){
        this->Projected_Omegas[i].clear();
        assert(this->Projected_Omegas[i].size() == 0);
        for (int j=0; j<tensor_mode_lengths[i]; j++){
          this->Projected_Omegas[i].push_back(0);
        }
        assert(this->Projected_Omegas[i].size() == this->tensor_mode_lengths[i]);
      }
      uint64_t ntensor_elements = 1;
      for (int i=0; i<this->tensor_mode_lengths.size(); i++){
          ntensor_elements *= this->tensor_mode_lengths[i];
      }

      std::vector<int64_t> training_nodes;
      std::vector<double> training_node_data;
      std::map<std::vector<int>,double> node_data_dict;
      std::map<std::vector<int>,int> node_count_dict;
      int configuration_offset = 0;
      //NOTE: This loop structure below is most efficient when provided data has already been averaged and thus no duplicate configurations are present.
      for (int i=0; i<num_configurations; i++){
          std::vector<int> element_key;
          bool is_valid = true;
          for (int j=0; j<this->order; j++){
            assert(configurations[i*this->order+j] >= this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]]);
            assert(configurations[i*this->order+j] <= this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]+this->tensor_mode_lengths[j]-((this->param_types[j]==NUMERICAL && this->interval_spacing[j]!=SINGLE) ? 0 : 1)]);
            if (this->param_types[j]!= NUMERICAL){
              element_key.push_back(get_node_index(configurations[i*this->order+j],this->tensor_mode_lengths[j],&this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]],this->interval_spacing[j]));
            } else{
              assert(this->tensor_mode_lengths[j]>0);
              element_key.push_back(get_interval_index(configurations[i*this->order+j],this->tensor_mode_lengths[j]+(this->interval_spacing[j]==SINGLE ? 0 : 1),&this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]],this->fixed_interval_spacing[j]));
              if (this->fixed_interval_spacing[j]==SINGLE){
                assert(this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]+element_key[j]] == configurations[i*this->order+j]);
              } else{
                assert(this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]+element_key[j]] <= configurations[i*this->order+j]);
                assert(this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]+element_key[j]+1] >= configurations[i*this->order+j]);
              }
            }
            assert(element_key[j] >= 0 && element_key[j] < this->tensor_mode_lengths[j]);
          }
          if (node_data_dict.find(element_key) == node_data_dict.end()){
              node_count_dict[element_key] = 1;
              node_data_dict[element_key] = runtimes[i];
          } else{
              node_count_dict[element_key]++;
              node_data_dict[element_key] += runtimes[i];
          }
      }
      double density = node_data_dict.size()*1./ntensor_elements;
      for (auto it : node_count_dict){
          int64_t tensor_elem = 0;
          int64_t scale=1;
          for (int i=0; i<this->order; i++){
            tensor_elem += (scale*it.first[i]);
            scale *= this->tensor_mode_lengths[i]; 
            assert(i<this->Projected_Omegas.size());
            assert(it.first[i] < this->Projected_Omegas[i].size());
            this->Projected_Omegas[i][it.first[i]]++;
          }
          assert(tensor_elem < ntensor_elements);
          assert(tensor_elem >=0);
          training_nodes.push_back(tensor_elem);
          training_node_data.push_back(node_data_dict[it.first]*1./it.second);
          assert(training_node_data[training_node_data.size()-1] > 0);
      }
      this->timer3 += (get_wall_time() - start_time);
      if (compute_fit_error){
        loss[0]=density;
        loss[1]=ntensor_elements;
      }
      start_time = get_wall_time();

      std::vector<double> ones(training_nodes.size(),1.);
      std::vector<int> tensor_mode_types(this->order,NS);
      int64_t num_nnz_elems,num_nnz_elems2,num_nnz_elems3;
      int64_t* ptr_to_indices,*ptr_to_indices2,*ptr_to_indices3;
      double* ptr_to_data,*ptr_to_data2,*ptr_to_data3;
      // NOTE: Not interested in doing any distributed-memory tensor contractions here, as this must proceed online.
      CTF::World dw(cm_training);
      CTF::Tensor<> omega(this->order, true, &this->tensor_mode_lengths[0], &tensor_mode_types[0], dw);
      CTF::Tensor<> Tsparse(this->order, true, &this->tensor_mode_lengths[0], &tensor_mode_types[0], dw);
      omega.write(training_nodes.size(),&training_nodes[0],&ones[0]);
      Tsparse.write(training_nodes.size(),&training_nodes[0],&training_node_data[0]);

      std::vector<CTF::Tensor<>*> FM1;// Must be Tensor, not Matrix due to constraints of multilinear interface
      std::vector<CTF::Tensor<>*> FM2;
      std::vector<CTF::Tensor<>*> FM2_svd;
      std::vector<int> fm_mode_types(2,NS);
      for (int i=0; i<this->order; i++){
          std::vector<int> fm_mode_lengths = {this->cp_rank[0],this->tensor_mode_lengths[i],this->cp_rank[1],this->tensor_mode_lengths[i]};
          FM1.emplace_back(new CTF::Tensor<>(2,&fm_mode_lengths[0],&fm_mode_types[0],dw));
          FM2.emplace_back(new CTF::Tensor<>(2,&fm_mode_lengths[2],&fm_mode_types[0],dw));
          init_factor_matrix(FM1[FM1.size()-1],loss_function);
          init_factor_matrix(FM2[FM2.size()-1],"MLogQ2");
       }
       // Optimize model

       // For interpolation, we first minimize mean squared error using log-transformed data
       auto _T_ = Tsparse;
       if (this->response_transform==1){
           _T_.get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,true);
           assert(num_nnz_elems == training_nodes.size());
           for (int j=0; j<num_nnz_elems; j++) ptr_to_data[j] = log(ptr_to_data[j]);
           _T_.write(num_nnz_elems,ptr_to_indices,ptr_to_data);
           delete[] ptr_to_data;
           delete[] ptr_to_indices;
       }
      this->timer4 += (get_wall_time() - start_time);
      start_time = get_wall_time();

      int max_num_re_inits = MAX_NUM_RE_INITS;
      char* env_var_ptr_reinit = std::getenv("CPPM_MAX_NUM_RE_INITS");
      if (env_var_ptr_reinit != NULL) max_num_re_inits = atoi(env_var_ptr_reinit);
      double re_init_loss_tol = RE_INIT_LOSS_TOL;
      env_var_ptr_reinit = std::getenv("CPPM_RE_INIT_LOSS_TOL");
      if (env_var_ptr_reinit != NULL) re_init_loss_tol = atof(env_var_ptr_reinit);
      //TODO: Use strcpy
      if (std::string(loss_function) == "MSE"){
        loss[3+this->order] = cpd_als(&dw,&_T_,&omega,FM1,regularization[0],model_convergence_tolerance[0],maximum_num_sweeps[0]);
      } else if (std::string(loss_function) == "MLogQ2"){
         int num_re_inits = 0;
         while (num_re_inits < max_num_re_inits){
           loss[3+this->order] = cpd_amn(&dw,&Tsparse,&omega,FM1,regularization[0],\
             model_convergence_tolerance[0],maximum_num_sweeps[0], factor_matrix_convergence_tolerance,\
             maximum_num_iterations, barrier_range[1],barrier_range[0],barrier_reduction_factor);
           if (loss[3+this->order] <= re_init_loss_tol) break;
           for (int i=0; i<this->order; i++) init_factor_matrix(FM1[i],loss_function);
           num_re_inits++;
         }
      } else assert(0);
      this->timer5 += (get_wall_time() - start_time);
      start_time = get_wall_time();
       int num_newton_iter1 = 0;
       loss[4+this->order] = 0;
       // For extrapolation, we minimize MLogQ2
       _T_ = Tsparse;// Re-copy because we want (the original) non-log-transformed elements

       int num_re_inits = 0;
       while (num_re_inits < max_num_re_inits){
         loss[4+this->order] = cpd_amn(&dw,&_T_,&omega,FM2,regularization[1],\
           model_convergence_tolerance[1],maximum_num_sweeps[1], factor_matrix_convergence_tolerance,\
           maximum_num_iterations, barrier_range[1],barrier_range[0],barrier_reduction_factor);
         if (loss[4+this->order] <= re_init_loss_tol) break;
         for (int i=0; i<this->order; i++) init_factor_matrix(FM2[i],loss_function);
         num_re_inits++;
       }  
      this->timer6 += (get_wall_time() - start_time);
      start_time = get_wall_time();

           std::vector<std::vector<double>> valid_tensor_cell_points(this->order);
           int num_numerical_fm_rows=0;
           for (int i=0; i<this->numerical_modes.size(); i++){
             num_numerical_fm_rows += this->tensor_mode_lengths[this->numerical_modes[i]];
           }
           double* temporary_extrap_models = new double[(1+max_spline_degree+1+this->cp_rank[1])*num_numerical_fm_rows];
           num_numerical_fm_rows=0;//reset as index into temporary_extrap_models
           for (int i=0; i<this->order; i++){
               if (this->param_types[i] == CATEGORICAL){
                   FM2_svd.push_back(nullptr);
                   FM2_svd.push_back(nullptr);
                   FM2_svd.push_back(nullptr);
                   continue;
               }
               // Inspect the FM to identify the projected set size
               int local_projected_set_size = 0;
               int max_elem = *std::max_element(this->Projected_Omegas[i].begin(),this->Projected_Omegas[i].end());
               for (int j=0; j<FM2[i]->lens[1]; j++){
                   assert(j<this->Projected_Omegas[i].size());
                   if (this->Projected_Omegas[i][j] >= min(projection_set_size_threshold_[i],max_elem)){
                       local_projected_set_size++;
                       if (this->param_types[i]==SINGLE){
                         valid_tensor_cell_points[i].push_back(this->tensor_cell_nodes[j]);
                       } else{
                         assert(!(j+1 == this->tensor_mode_lengths[i]+1));
                         valid_tensor_cell_points[i].push_back(get_midpoint_of_two_nodes(j,this->tensor_mode_lengths[i]+1,&this->tensor_cell_nodes[this->tensor_cell_node_offsets[i]],this->interval_spacing[i],30));
                       }
                       assert(valid_tensor_cell_points[i][local_projected_set_size-1]>0);
                   }
               }
               assert(local_projected_set_size>0);
               assert(local_projected_set_size>=this->cp_rank[1] );
               if (this->cp_rank[1] > local_projected_set_size){
                 // Simply don't extrapolate
                 FM2_svd.push_back(nullptr);
                 FM2_svd.push_back(nullptr);
                 FM2_svd.push_back(nullptr);
                 continue;
               }
               assert(1+max_spline_degree <= local_projected_set_size);
               //NOTE: notice below that we switch the order of mode lengths for effective use of LAPACK
               CTF::Matrix<> Factor_Matrix__(local_projected_set_size,this->cp_rank[1],dw);
               Factor_Matrix__.get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data);
               FM2[i]->get_local_data(&num_nnz_elems2,&ptr_to_indices2,&ptr_to_data2);
               assert(num_nnz_elems <= num_nnz_elems2);
               int column_count = 0;
               for (int j=0; j<FM2[i]->lens[1]; j++){
                   assert(j<this->Projected_Omegas[i].size());
                   if (this->Projected_Omegas[i][j] >= min(projection_set_size_threshold_[i],max_elem)){
                       for (int k=0; k<FM2[i]->lens[0]; k++){
                           ptr_to_data[column_count+local_projected_set_size*k] = ptr_to_data2[k+FM2[i]->lens[0]*j];
                       }
                       column_count++;
                   }
               }
/*
               if (world_rank==0){
                 std::cout << "Extrapolation with factor matrix " << i << "\n";
                 for (int j=0; j<num_nnz_elems2; j++) std::cout << ptr_to_data2[j] << " ";
                 std::cout << "\n";
                 for (int j=0; j<num_nnz_elems; j++) std::cout << ptr_to_data[j] << " ";
                 std::cout << "\n";
               }
*/
               //NOTE: Factor_Matrix__ has data stored column-wise
               Factor_Matrix__.write(num_nnz_elems,ptr_to_indices,ptr_to_data);
               delete[] ptr_to_data;
               delete[] ptr_to_data2;
               delete[] ptr_to_indices;
               delete[] ptr_to_indices2;
               // Prep for a rank-1 SVD -- NOTE: We store all relevant columns/rows, not just low-rank factorization
               FM2_svd.push_back(new CTF::Matrix<>(local_projected_set_size,this->cp_rank[1],dw));
               FM2_svd.push_back(new CTF::Vector<>(this->cp_rank[1],dw));
               FM2_svd.push_back(new CTF::Matrix<>(this->cp_rank[1],this->cp_rank[1],dw));
               CTF::Matrix<>* left_singular_matrix_ptr = dynamic_cast<CTF::Matrix<>*>(FM2_svd[3*i]);
               CTF::Vector<>* singular_value_ptr = dynamic_cast<CTF::Vector<>*>(FM2_svd[3*i+1]);
               CTF::Matrix<>* right_singular_matrix_ptr = dynamic_cast<CTF::Matrix<>*>(FM2_svd[3*i+2]);
               Factor_Matrix__.svd(*left_singular_matrix_ptr,*singular_value_ptr,*right_singular_matrix_ptr,1);
               FM2_svd[3*i]->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data);
               FM2_svd[3*i+1]->get_local_data(&num_nnz_elems3,&ptr_to_indices3,&ptr_to_data3);
               FM2_svd[3*i+2]->get_local_data(&num_nnz_elems2,&ptr_to_indices2,&ptr_to_data2);
               // Identify Perron vector: simply flip signs if necessary (of the first column only!)
/*
               if (world_rank==0){
                 std::cout << "Extrapolation with factor matrix " << i << ", which has size " << this->tensor_mode_lengths[i] << " " << this->cp_rank[1] << std::endl;
                 for (int j=0; j<this->tensor_mode_lengths[0]; j++) std::cout << this->Projected_Omegas[0][j] << " ";
                 std::cout << "\n";
                 for (int j=0; j<this->tensor_mode_lengths[1]; j++) std::cout << this->Projected_Omegas[1][j] << " ";
                 std::cout << "\n";
                 for (int j=0; j<num_nnz_elems; j++) std::cout << ptr_to_data[j] << " ";
                  std::cout << "\n";
                 for (int j=0; j<num_nnz_elems3; j++) std::cout << ptr_to_data3[j] << " ";
                  std::cout << "\n";
                 for (int j=0; j<num_nnz_elems2; j++) std::cout << ptr_to_data2[j] << " ";
                  std::cout << "\n";
               }
*/
               // Curate the Perron vector: make it positive (which is always possible, see Thereom), but also restrict to constructing model solely out of strictly-increasing elements
               int num_elements_to_keep = local_projected_set_size;// should be <= local_projected_set_size
               for (int j=0; j<local_projected_set_size; j++){
                   if (ptr_to_data[j]<0) ptr_to_data[j] *= (-1);
               }
               for (int j=0; j<this->cp_rank[1]; j++){
                   //TODO: Note, this may be wrong due to how VT/V is accessed.
                   //if (ptr_to_data2[j]<0) ptr_to_data2[j] *= (-1);
                   if (ptr_to_data2[j*this->cp_rank[0]]<0) ptr_to_data2[j*this->cp_rank[0]] *= (-1);
               }
               assert(ptr_to_data3[0]>0);
               assert(num_elements_to_keep <= local_projected_set_size);
               //FM2_svd[3*i]->write(num_nnz_elems,ptr_to_indices,ptr_to_data);
               //FM2_svd[3*i+2]->write(num_nnz_elems2,ptr_to_indices2,ptr_to_data2);

               // Create simple linear model from ptr_to_data
               std::vector<double> feature_matrix(num_elements_to_keep*(1+max_spline_degree),1.);
               for (int j=1; j<=max_spline_degree; j++){
                 for (int k=0; k<num_elements_to_keep; k++){
                   //double midpoint_val = get_midpoint_of_two_nodes(k,local_projected_set_size+1,&this->tensor_cell_nodes[this->tensor_cell_node_offsets[i]],this->interval_spacing[i]);
                   assert(k<valid_tensor_cell_points[i].size());
                   double point_val = valid_tensor_cell_points[i][k];
                   assert(point_val>0);
                   feature_matrix[num_elements_to_keep*j+k] = log(point_val)*feature_matrix[num_elements_to_keep*(j-1)+k];
                 }
               }
               for (int k=0; k<num_elements_to_keep; k++){
                 ptr_to_data[k] = log(ptr_to_data[k]);
               }
               //For simplicity sake, just solve for model coefficients via SVD
               int info;
               std::vector<double> work_buffer(1,-1);
               CTF_LAPACK::cdgels('N',num_elements_to_keep,1+max_spline_degree,1,&feature_matrix[0],num_elements_to_keep,&ptr_to_data[0],num_elements_to_keep,&work_buffer[0],-1,&info);
               assert(work_buffer[0]>0);
               int lwork = work_buffer[0];
               work_buffer.resize(lwork);
               CTF_LAPACK::cdgels('N',num_elements_to_keep,1+max_spline_degree,1,&feature_matrix[0],num_elements_to_keep,&ptr_to_data[0],num_elements_to_keep,&work_buffer[0],lwork,&info);
               assert(info==0);
               int jump = 1+max_spline_degree+1+this->cp_rank[1];
               // Write model coefficients that fit the single left-singular vector
               std::memcpy(&temporary_extrap_models[num_numerical_fm_rows*jump],&ptr_to_data[0],sizeof(double)*(1+max_spline_degree));
               // Write single singular value
               std::memcpy(&temporary_extrap_models[num_numerical_fm_rows*jump+(1+max_spline_degree)],&ptr_to_data3[0],sizeof(double));
               // Write single row of VT
               std::memcpy(&temporary_extrap_models[num_numerical_fm_rows*jump+(1+max_spline_degree+1)],&ptr_to_data2[0],this->cp_rank[1]*sizeof(double));
               num_numerical_fm_rows++;
               delete[] ptr_to_data;
               delete[] ptr_to_data2;
               delete[] ptr_to_data3;
               delete[] ptr_to_indices;
               delete[] ptr_to_indices2;
               delete[] ptr_to_indices3;
           }
/* TODO: Fill these in, change to using a simple cubic (4-term) model (for now), so no splines. Extract only first column.
           for (int i=0; i<this->order; i++){//One cannot simply extrapolate only certain modes.
               if (this->param_types[i] == CATEGORICAL){
                   extrap_models.append([]);
                   continue;
               }
               extrap_models.append(pe.Earth(max_terms=10,max_degree=max_spline_degree,allow_linear=True));
               //extrap_models[-1].fit(np.log(parameter_nodes[i].reshape(-1,1)),np.log(FM2_sv[i][0]));
               print(np.array(updated_parameter_nodes[i]).reshape(-1,1),FM2_sv[i][0]);
               extrap_models[-1].fit(np.log(np.array(updated_parameter_nodes[i]).reshape(-1,1)),np.log(FM2_sv[i][0]));
           }
*/
      this->timer7 += (get_wall_time() - start_time);
      start_time = get_wall_time();
      //NOTE: Factor matrices are written as: tensor mode grows slowest, row of FM grows fastest

      int total_elements_across_all_factor_matrices = 0;
      for (int i=0; i<this->order; i++){
        total_elements_across_all_factor_matrices += this->tensor_mode_lengths[i];
      }
      total_elements_across_all_factor_matrices *= this->cp_rank[0];
      double* temporary_buffer = new double[total_elements_across_all_factor_matrices];
      int offset = 0;
      for (int i=0; i<this->order; i++){
        FM1[i]->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data);
        assert(this->tensor_mode_lengths[i]*this->cp_rank[0] == num_nnz_elems);
        std::memcpy(temporary_buffer+offset,ptr_to_data,num_nnz_elems*sizeof(double));
        offset += num_nnz_elems;
        delete[] ptr_to_data;
        delete[] ptr_to_indices;
      }
      if (this->factor_matrix_elements_generalizer!=nullptr) delete[] this->factor_matrix_elements_generalizer;
      this->factor_matrix_elements_generalizer = temporary_buffer;
      total_elements_across_all_factor_matrices = 0;
      for (int i=0; i<this->order; i++){
        total_elements_across_all_factor_matrices += this->tensor_mode_lengths[i];
      }
      total_elements_across_all_factor_matrices *= this->cp_rank[1];
      temporary_buffer = new double[total_elements_across_all_factor_matrices];
      offset = 0;
      for (int i=0; i<this->order; i++){
        FM2[i]->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data);
        assert(this->tensor_mode_lengths[i]*this->cp_rank[1] == num_nnz_elems);
        std::memcpy(temporary_buffer+offset,ptr_to_data,num_nnz_elems*sizeof(double));
        offset += num_nnz_elems;
        delete[] ptr_to_data;
        delete[] ptr_to_indices;
      }
      if (this->factor_matrix_elements_extrapolator!=nullptr) delete[] this->factor_matrix_elements_extrapolator;
      this->factor_matrix_elements_extrapolator = temporary_buffer;
      if (this->factor_matrix_elements_extrapolator_models != nullptr) delete[] this->factor_matrix_elements_extrapolator_models;
      this->factor_matrix_elements_extrapolator_models = temporary_extrap_models;
      for (int i=0; i<this->order; i++){
        delete FM1[i];
        delete FM2[i];
      }
      for (int i=0; i<FM2_svd.size(); i++){
        if (FM2_svd[i] != nullptr) delete FM2_svd[i];
      }

      this->is_valid=true;
      if (compute_fit_error){
        double agg_relative_error=0;
        double agg_quadrature_error=0;
        double agg_low_rank_error=0;

        bool do_i_want_to_print_out = false;//world_rank==0 && this->tensor_mode_lengths[0]>10 && this->tensor_mode_lengths[1]>10;
        if (do_i_want_to_print_out){
          std::cout << "Start of training set prediction\n";
          std::cout << this->param_types[0] << " " << this->param_types[1] << "\n";
          std::cout << this->interval_spacing[0] << " " << this->interval_spacing[1] << "\n";
          std::cout << this->tensor_mode_lengths[0] << " " << this->tensor_mode_lengths[1] << "\n";
          std::cout << this->tensor_cell_node_offsets[0] << " " << this->tensor_cell_node_offsets[1] << "\n";
          for (int i=0; i<this->tensor_cell_nodes.size(); i++){
            std::cout << this->tensor_cell_nodes[i] << " ";
          }
          std::cout << "\n";
        }

        for (int i=0; i<num_configurations; i++){
          if (do_i_want_to_print_out) std::cout << "Predicting from training set (" << configurations[i*this->order] << "," << configurations[i*this->order+1] << std::endl;
          double runtime_estimate = this->predict(&configurations[i*this->order]);
          double rel_err = runtime_estimate<=0 ? 1 : fabs(log(runtime_estimate/runtimes[i]));
          if (do_i_want_to_print_out) std::cout << "Runtime estimate - " << runtime_estimate << ", true runtime - " << runtimes[i] << ", rel_err - " << rel_err << std::endl;
          std::vector<int> element_key;
          std::vector<double> element_key_configuration;
          for (int j=0; j<this->order; j++){
            assert(configurations[i*this->order+j] <= this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]+this->tensor_mode_lengths[j]-(this->param_types[j]==NUMERICAL && this->interval_spacing[j]!=SINGLE ? 0 : 1)]);
            if (this->param_types[j]!= NUMERICAL){
              element_key.push_back(get_node_index(configurations[i*this->order+j],this->tensor_mode_lengths[j],&this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]],this->interval_spacing[j]));
            } else{
              assert(this->tensor_mode_lengths[j]>0);
              element_key.push_back(get_interval_index(configurations[i*this->order+j],this->tensor_mode_lengths[j]+(this->interval_spacing[j]==SINGLE ? 0 : 1),&this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]],this->fixed_interval_spacing[j]));
              if (this->fixed_interval_spacing[j]==SINGLE){
                assert(this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]+element_key[j]] == configurations[i*this->order+j]);
              } else{
                assert(this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]+element_key[j]] <= configurations[i*this->order+j]);
                assert(this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]+element_key[j]+1] >= configurations[i*this->order+j]);
              }
            }
            assert(element_key[j] >= 0 && element_key[j] < this->tensor_mode_lengths[j]);
            if (this->param_types[j]==CATEGORICAL){
              element_key_configuration.push_back(this->tensor_cell_nodes[element_key[j]]);
            } else{
              if (!this->interval_spacing[j]==SINGLE) assert(!(element_key[j]+1==this->tensor_mode_lengths[j]+1));
              element_key_configuration.push_back(this->interval_spacing[j]==SINGLE ? this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]+element_key[j]] : get_midpoint_of_two_nodes(element_key[j],this->tensor_mode_lengths[j]+1,&this->tensor_cell_nodes[this->tensor_cell_node_offsets[j]],this->interval_spacing[j],41));
            }
          }
          assert(node_data_dict.find(element_key) != node_data_dict.end());
          assert(node_count_dict.find(element_key) != node_count_dict.end());
          double tensor_elem_sample_mean = node_data_dict[element_key] / node_count_dict[element_key];// use this rather than training_nodes because I don't care about calculating tensor_elem as one integer
          double quad_err = fabs(log(tensor_elem_sample_mean/runtimes[i]));
///*
          double tensor_elem_prediction = multilinear_product(&this->factor_matrix_elements_generalizer[0],&element_key[0],&this->tensor_mode_lengths[0],this->order,this->cp_rank[0]);
          if (this->response_transform==1) tensor_elem_prediction = exp(tensor_elem_prediction);
          double low_rank_err1 = fabs(log(tensor_elem_prediction/tensor_elem_sample_mean));
          double tensor_elem_prediction2 = this->predict(&element_key_configuration[0]);
          double low_rank_err2 = fabs(log(tensor_elem_prediction2/tensor_elem_sample_mean));
/*
          if ((tensor_elem_prediction2-tensor_elem_prediction)>1e-6){
            for (int j=0; j<this->tensor_cell_nodes.size(); j++) std::cout << this->tensor_cell_nodes[j] << " ";
            std::cout << "\n" << this->tensor_cell_node_offsets[0] << " " << this->tensor_cell_node_offsets[1] << " " << this->tensor_cell_node_offsets[2] << "\n";
            std::cout << configurations[i*this->order] << " " << configurations[i*this->order+1] << " " << configurations[i*this->order+2] << " " << element_key[0] << " " << element_key[1] << " " << element_key[2] << " " << element_key_configuration[0] << " " << element_key_configuration[1] << " " << element_key_configuration[2] << " " << tensor_elem_prediction << " " << tensor_elem_prediction2 << std::endl;
          }
*/
          assert((tensor_elem_prediction2-tensor_elem_prediction)<=1e-6); 
          if (do_i_want_to_print_out && quad_err <= .1 && low_rank_err1 <= .1 && rel_err >= .5){
            // get nearby predictions too (just for CTF)
            std::vector<int> element_key_left = {element_key[0],std::max(0,element_key[1]-1)};
            std::vector<int> element_key_right = {element_key[0],std::min(this->tensor_mode_lengths[1],element_key[1]+1)};
            std::vector<double> element_key_config_left = {element_key_configuration[0],get_midpoint_of_two_nodes(element_key_left[1],this->tensor_mode_lengths[1]+1,&this->tensor_cell_nodes[this->tensor_cell_node_offsets[1]],this->interval_spacing[1],41)};
            std::vector<double> element_key_config_right = {element_key_configuration[0],get_midpoint_of_two_nodes(element_key_right[1],this->tensor_mode_lengths[1]+1,&this->tensor_cell_nodes[this->tensor_cell_node_offsets[1]],this->interval_spacing[1],41)};
            double tensor_elem_left_prediction3 = this->predict(&element_key_config_left[0]);
            double tensor_elem_right_prediction3 = this->predict(&element_key_config_right[0]);
	    std::cout << "(" << configurations[i*this->order] << "," << configurations[i*this->order+1] << "), Node: (" << element_key[0] << "," << element_key[1] << "), Element config - (" << element_key_configuration[0] << "," << element_key_configuration[1] << "), Left: - " << tensor_elem_left_prediction3 << ", Right: " << tensor_elem_right_prediction3 << ", " << this->Projected_Omegas[1][element_key_left[1]] << "," << this->Projected_Omegas[1][element_key[1]] << "," << this->Projected_Omegas[1][element_key_right[1]] << "  , True Runtime: " << runtimes[i] << ", Runtime Preditions: (" << tensor_elem_sample_mean << " , " << tensor_elem_prediction << " , " << tensor_elem_prediction2 << " , " << runtime_estimate << "), Errors: (" << quad_err << " , " << low_rank_err1 << " , " << low_rank_err2 << " , " << rel_err << ")\n";
          }
          //NOTE: I could also check correlation between high relative error and high quadrature error
          agg_relative_error += rel_err;
          agg_quadrature_error += quad_err;
        }
        if (do_i_want_to_print_out){
          std::cout << "\n\n\n\n\n";
        }
        for (auto& it : node_data_dict){
          auto& element_key = it.first;
          double tensor_elem_sample_mean = node_data_dict[element_key] / node_count_dict[element_key];// use this rather than training_nodes because I don't care about calculating tensor_elem as one integer
          double tensor_elem_prediction = multilinear_product(&this->factor_matrix_elements_generalizer[0],&element_key[0],&this->tensor_mode_lengths[0],this->order,this->cp_rank[0]);
          if (this->response_transform==1) tensor_elem_prediction = exp(tensor_elem_prediction);
          double low_rank_err = fabs(log(tensor_elem_prediction/tensor_elem_sample_mean));
          //NOTE: I could also check correlation between high relative error and high quadrature error
          agg_low_rank_error += low_rank_err;
        }
        agg_relative_error /= num_configurations;
        agg_quadrature_error /= num_configurations;
        agg_low_rank_error /= node_data_dict.size();
        loss[5+this->order]=agg_relative_error;
        loss[6+this->order]=agg_quadrature_error;
        loss[7+this->order]=agg_low_rank_error;//TODO: NOTE: This is over-counting cells with more observations than others, but should be ok for now
      }
      this->interval_spacing = save_interval_spacing;// revert back if user wants to re-train
      if (delete_data){
        delete[] configurations;
        delete[] runtimes;
      }
      return true;
}
