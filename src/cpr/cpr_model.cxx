#include <cstring>
#include <cassert>
#include <utility>
#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <time.h>
#include <sys/time.h>

#include "cpr_model.h"
#include "cpr_types.h"
#include "cpr_hyperparameter_pack.h"
#include "cpr_parameter_pack.h"
#include "../types.h"
#include "ctf.hpp"
#include "../util.h"

#define NUMERICAL_PARAM_MIN_OBS_RANGE 32
#define MIN_POS_RUNTIME 1e-8  // Essentially a "zero" runtime
#define AMN_RESET 1e-6

namespace performance_model{

template<typename DataType>
void sparse_add(CTF::Tensor<DataType>* T, CTF::Tensor<DataType>* M, double alpha, double beta){
  IASSERT(T->order == M->order);
  IASSERT(T->is_sparse && M->is_sparse);

  int64_t npair1,npair2;
  CTF::Pair<DataType>* pairs1;
  CTF::Pair<DataType>* pairs2;

  npair1 = T->nnz_loc;
  npair2 = M->nnz_loc;
  IASSERT(npair1==npair2);
  for (int i=0; i<T->order; i++){
    IASSERT(T->edge_map[i].calc_phys_phase() == M->edge_map[i].calc_phys_phase());
  }
  pairs1 = (CTF::Pair<DataType>*)T->data;
  pairs2 = (CTF::Pair<DataType>*)M->data;

  for(int64_t i=0;i<npair1;i++){
     pairs1[i].d = alpha*pairs1[i].d + beta*pairs2[i].d;
  }
}

template<typename DataType>
void sparse_mul(CTF::Tensor<DataType>* T, CTF::Tensor<DataType>* M){
  IASSERT(T->order == M->order);
  IASSERT(T->is_sparse && M->is_sparse);

  int64_t npair1,npair2;
  CTF::Pair<DataType>* pairs1;
  CTF::Pair<DataType>* pairs2;

  npair1 = T->nnz_loc;
  npair2 = M->nnz_loc;
  IASSERT(npair1==npair2);
  for (int i=0; i<T->order; i++){
    IASSERT(T->edge_map[i].calc_phys_phase() == M->edge_map[i].calc_phys_phase());
  }
  pairs1 = (CTF::Pair<DataType>*)T->data;
  pairs2 = (CTF::Pair<DataType>*)M->data;

  for(int64_t i=0;i<npair1;i++){
     pairs1[i].d *= pairs2[i].d;
  }
}

template<typename DataType>
void sparse_exp(CTF::Tensor<DataType>* T){
  IASSERT(T->is_sparse);

  int64_t npair;
  CTF::Pair<DataType>* pairs;
  npair = T->nnz_loc;
  pairs = (CTF::Pair<DataType>*)T->data;
  for(int64_t i=0;i<npair;i++){
     pairs[i].d = std::exp(pairs[i].d);
  }
}

template<typename DataType>
void sparse_log(CTF::Tensor<DataType>* T){
  IASSERT(T->is_sparse);

  int64_t npair;
  CTF::Pair<DataType>* pairs;
  npair = T->nnz_loc;
  pairs = (CTF::Pair<DataType>*)T->data;
  for(int64_t i=0;i<npair;i++){
    pairs[i].d = std::log(pairs[i].d);
  }
}

template<typename DataType>
void get_index_tensor(CTF::Tensor<DataType>* T){
  IASSERT(T->is_sparse);
  int64_t npair;
  CTF::Pair<DataType>* pairs;
  npair = T->nnz_loc;
  pairs = (CTF::Pair<DataType>*)T->data;
  for(int64_t i=0;i<npair;i++){
    pairs[i].d = 1.0;
  }
}

template<typename DataType>
void sparse_inv(CTF::Tensor<DataType>* T){
  IASSERT(T->is_sparse);
  int64_t npair = T->nnz_loc;
  CTF::Pair<DataType>* pairs = (CTF::Pair<DataType>*)T->data;
  for(int64_t i=0;i<npair;i++){
     pairs[i].d = 1./pairs[i].d;
  }
}

template<typename DataType>
void sparse_add1(CTF::Tensor<DataType>* T){
  IASSERT(T->is_sparse);
  int64_t npair = T->nnz_loc;
  CTF::Pair<DataType>* pairs = (CTF::Pair<DataType>*)T->data;
  for(int64_t i=0;i<npair;i++){
     pairs[i].d = 1.+pairs[i].d;
  }
}

template<typename DataType>
void sparse_copy(CTF::Tensor<DataType>* T1, CTF::Tensor<DataType>* T2){
  IASSERT(T1->is_sparse);
  IASSERT(T2->is_sparse);
  int64_t npair1 = T1->nnz_loc;
  CTF::Pair<DataType>* pairs1 = (CTF::Pair<DataType>*)T1->data;
  int64_t npair2 = T2->nnz_loc;
  CTF::Pair<DataType>* pairs2 = (CTF::Pair<DataType>*)T2->data;
  IASSERT(npair1==npair2);
  for(int64_t i=0;i<npair1;i++){
     pairs1[i].d = pairs2[i].d;
  }
}

template<typename DataType>
void sparse_neg(CTF::Tensor<DataType>* T){
  IASSERT(T->is_sparse);
  int64_t npair = T->nnz_loc;
  CTF::Pair<DataType>* pairs = (CTF::Pair<DataType>*)T->data;
  for(int64_t i=0;i<npair;i++){
     pairs[i].d = (-1.)*pairs[i].d;
  }
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

void init_factor_matrix(CTF::Tensor<>* fm, loss_function t_loss){
  if (fm==nullptr) return;
  //assert(fm != nullptr);
  int cp_rank = fm->lens[0];
  int mode_length = fm->lens[1];
  if (t_loss == loss_function::MSE){
    fm->fill_random(0,1);
    // Enforce to be strictly increasing
    int64_t num_nnz_elems;
    int64_t* ptr_to_indices;
    double* ptr_to_data;
    fm->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,true);
    for (int k=0; k<cp_rank; k++){
      for (int j=1; j<mode_length; j++){
        ptr_to_data[j*cp_rank+k] += ptr_to_data[(j-1)*cp_rank+k];
      }
      for (int j=0; j<mode_length; j++){
        ptr_to_data[j*cp_rank+k] -= 1;
      }
    }
    fm->write(num_nnz_elems,ptr_to_indices,ptr_to_data);
    delete[] ptr_to_data;
    delete[] ptr_to_indices;
  }
  else if (t_loss == loss_function::MLOGQ2){
    fm->fill_random(0,1);
    // Enforce to be strictly increasing
    int64_t num_nnz_elems;
    int64_t* ptr_to_indices;
    double* ptr_to_data;
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
}

void debug_tensor(CTF::Tensor<>* t, const std::string& str){
  int64_t num_nnz_elems;
  int64_t* ptr_to_indices;
  double* ptr_to_data;
  t->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,false);
  std::cout << str << "  ";
  for (int j=0; j<num_nnz_elems; j++) std::cout << ptr_to_data[j] << " ";
  std::cout << "\n";
  delete[] ptr_to_data;
  delete[] ptr_to_indices;
}

// Normalize each column within a factor matrix
void normalize(std::vector<CTF::Tensor<>*>& X){
  // Re-normalize by iterating over each column of each factor matrix
  int order = X.size();
  int rank = X[0]->lens[0];// choice of index 0 is arbitrary
  for (int j=0; j<rank; j++){
    double weight = 1;
    std::vector<int> offsets1 = {j,0};
    std::vector<int> offsets2 = {0,0};
    // Iterate over the j'th column of all d factor matrices
    for (int k=0; k<order; k++){
      std::vector<int> ends1 = {j+1,static_cast<int>(X[k]->lens[1])};
      std::vector<int> ends2 = {1,static_cast<int>(X[k]->lens[1])};
      CTF::Tensor<> temp = X[k]->slice(&offsets1[0],&ends1[0]);
      double nrm = temp.norm2();
      weight *= nrm;
      X[k]->slice(&offsets1[0],&ends1[0],1./nrm,temp,&offsets2[0],&ends2[0],0);
    }
    weight = pow(weight,1./order);
    for (int k=0; k<order; k++){
      std::vector<int> ends1 = {j+1,static_cast<int>(X[k]->lens[1])};
      std::vector<int> ends2 = {1,static_cast<int>(X[k]->lens[1])};
      CTF::Tensor<> temp = X[k]->slice(&offsets1[0],&ends1[0]);
      X[k]->slice(&offsets1[0],&ends1[0],weight,temp,&offsets2[0],&ends2[0],0);
    }
  }
}

struct MLogQ2{
    //Current implementation is using \lambda  = e^m and replacing it in the function to get: e^m - xm
    static void Get_RHS(CTF::World* dw, CTF::Tensor<>* T, CTF::Tensor<>* O, std::vector<CTF::Tensor<>*>& A, int num, double reg, double mu, CTF::Tensor<>* grad, CTF::Tensor<>* Hessian){
        CTF::Tensor<> M(O);
        std::vector<int> mode_list(A.size());
        for (int j=0; j<A.size(); j++) mode_list[j]=j;
        CTF::TTTP(&M,A.size(),&mode_list[0],&A[0],true);
        CTF::Tensor<> M_reciprocal3 = *T;

        sparse_inv(&M);
        sparse_mul(&M_reciprocal3,&M);
        sparse_neg(&M);
        sparse_log(&M_reciprocal3);
        sparse_copy(Hessian,&M_reciprocal3);
        sparse_add1(Hessian);
        sparse_mul(&M_reciprocal3,&M);
        sparse_mul(&M,&M);
        sparse_mul(Hessian,&M);

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
        CTF::MTTKRP(&M_reciprocal3,&lst_mat[0],num,true);
        // Notice that grad should be negative, but it is not!
        //         This is taken into account when we subtract the step from the FM in 'step(..)'
        // Notice: no factors of 2. These are divided away automatically, as both the loss, regularization terms, etc. have them.
        CTF::Function<> gradient_update([&reg,&mu](double d1, double d2) -> double { return d1+reg*d2 - mu/2./d2; });
        (*grad)["ij"] = gradient_update((*lst_mat[num])["ij"],(*A[num])["ij"]);
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
                    //std::cout << i << ": step_nrm - " << step_nrm << std::endl;
                    prev_step_nrm = step_nrm;
                    (*A[i])["ij"] = (*A[i])["ij"] - (*lst_mat[i])["ij"];
                    (*lst_mat[i])["ij"] = (*A[i])["ij"];
                    // Verify that following update of factor matrix, every element is positive.
                    lst_mat[i]->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,false);
                    A[i]->get_local_data(&num_nnz_elems2,&ptr_to_indices2,&ptr_to_data2,false);
                    assert(num_nnz_elems == num_nnz_elems2);
                    for (int j=0; j<num_nnz_elems; j++){
                      if (ptr_to_data2[j] <= 0) ptr_to_data2[j]=AMN_RESET;
                    }
                    A[i]->write(num_nnz_elems2,ptr_to_indices2,ptr_to_data2);
                    lst_mat[i]->write(num_nnz_elems2,ptr_to_indices2,ptr_to_data2);
                    delete[] ptr_to_data;
                    delete[] ptr_to_data2;
                    delete[] ptr_to_indices;
                    delete[] ptr_to_indices2;
                    if (step_nrm <= factor_matrix_convergence_tolerance) break;
                }
                mu /= barrier_reduction_factor;
                newton_count += t;
            }
            delete lst_mat[i];
        }
        return newton_count;
    }
};

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
  CTF::Function<> gradient_update([&reg](double d1, double d2) -> double { return d1-reg*d2; });
  (*grad)["ij"] = gradient_update((*lst_mat[num])["ij"],(*A[num])["ij"]);
  delete lst_mat[num];
 }

    static void step(CTF::World* dw, CTF::Tensor<>* T, CTF::Tensor<>* O, std::vector<CTF::Tensor<>*>& A, double reg, int64_t nnz){
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
            (*A[i])["ij"] = (*lst_mat[i])["ij"];
            delete lst_mat[i];
        }
        // Updated factor matrix filled in via pointer, so return nothing
    }
};

double cpd_als(CTF::World* dw, CTF::Tensor<>* T_in, CTF::Tensor<>* O, std::vector<CTF::Tensor<>*> X, double reg, double model_convergence_tolerance, int max_nsweeps, loss_function t_loss = loss_function::MSE){
    assert(t_loss == loss_function::MSE);

    // X - model parameters, framed as a guess
    // O - sparsity pattern encoded as a sparse matrix
    // T_in - sparse tensor of data
    // Assumption is that the error is MSE with optional regularization
    int64_t nnz = O->nnz_loc;
    double err=100000000.;
    int n_newton_iterations=0;
    std::cout << "nnz - " << nnz << std::endl;
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

        sparse_add(&M,T_in,1,-1);
        err = M.norm2()/sqrt(nnz);
        err *= err;
        std::cout << "Loss: " << err << std::endl;
        if (err < model_convergence_tolerance) break;
    }
    return err;
}

double cpd_amn(CTF::World* dw, CTF::Tensor<>* T_in, CTF::Tensor<>* O, std::vector<CTF::Tensor<>*> X, double reg, double model_convergence_tolerance, int max_nsweeps, double factor_matrix_convergence_tolerance, int max_newton_iter, double barrier_start=1e1, double barrier_stop=1e-11, double barrier_reduction_factor=8, loss_function t_loss=loss_function::MLOGQ2){
    assert(t_loss == loss_function::MLOGQ2);
    int64_t nnz = O->nnz_loc;
    reg *= nnz;
    barrier_start *= nnz;
    barrier_stop *= nnz;
    double err=100000000.;
    double err_prev = 100000000.;


    int64_t num_nnz_elems;
    int64_t* ptr_to_indices;
    double* ptr_to_data;
    T_in->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,true);
    for (int k=0; k<num_nnz_elems; k++){
      //std::cout << ptr_to_data[k] << " ";
      assert(ptr_to_data[k]>0);
    }
    std::cout << "\n";
    delete[] ptr_to_data;
    delete[] ptr_to_indices;

/*
    X_prev = []
    for (int i=0; i<X.size(); i++){
        X_prev.append(X[i].copy());
    }
    double err=err_prev;
*/
    int n_newton_iterations=0;
    auto TT = *T_in;
    sparse_log(&TT);
    for (int i=0; i<max_nsweeps; i++){
      if (i>0){
        int _n_newton_iterations = MLogQ2::step(dw,T_in,O,X,reg,barrier_start,barrier_stop,barrier_reduction_factor,factor_matrix_convergence_tolerance, max_newton_iter);
        normalize(X);
        n_newton_iterations += _n_newton_iterations;
      }
/*
  for (int j=0; j<X.size(); j++){
    int64_t num_nnz_elems;
    int64_t* ptr_to_indices;
    double* ptr_to_data;
    X[j]->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data,false);
    std::cout << "Iteration " << i << ", FM " << j << ": ";
    for (int k=0; k<num_nnz_elems; k++) std::cout << ptr_to_data[k] << " ";
    std::cout << "\n";
    delete[] ptr_to_data;
    delete[] ptr_to_indices;
    }
*/
        CTF::Tensor<> M(O);
        std::vector<int> mode_list(X.size());
        for (int j=0; j<X.size(); j++) mode_list[j]=j;
        CTF::TTTP(&M,X.size(),&mode_list[0],&X[0],true);
        auto P = M;
        sparse_log(&P);

        sparse_add(&P,&TT,-1.,1.);
        err = P.norm2(); err*=err; err/=nnz;
        std::cout << "Loss: " << err << std::endl;
        if (err < model_convergence_tolerance) break;
        err_prev = err;
    }
    return err;
}

double get_midpoint_of_two_nodes(int idx, int num_nodes, const double* _nodes, parameter_range_partition node_spacing_type){
  // NOTE: 'idx' refers to the interval/partition, which reflects a single particular tensor element
  // NOTE: if node_spacing_type==parameter_range_partition::SINGLE, then the midpoint of two nodes doesn't reflect the execution time of an interval.
  // NOTE: if (parameter_range_partition::UNIFORM,parameter_range_partition::AUTOMATIC,parameter_range_partition::GEOMETRIC), 'idx' is assumed to be the leading coordinate of a cell. Therefore, 'idx'+1 is always valid
  if (node_spacing_type != parameter_range_partition::SINGLE) assert(idx < num_nodes-1);
  if (node_spacing_type == parameter_range_partition::GEOMETRIC || (node_spacing_type == parameter_range_partition::AUTOMATIC && _nodes[idx]>1)){
      assert(_nodes[idx+1] > _nodes[idx]);
      double scale = _nodes[idx+1]*1./_nodes[idx];
      assert(scale>0);
      double transf_scale = log(scale);
      return pow(scale,(log(_nodes[idx])/transf_scale + ((log(_nodes[idx+1])/transf_scale)-(log(_nodes[idx])/transf_scale))/2.));
  } else if (node_spacing_type == parameter_range_partition::UNIFORM || node_spacing_type == parameter_range_partition::AUTOMATIC){
      return _nodes[idx]*1. + (_nodes[idx+1]-_nodes[idx])/2.;
  } else if (node_spacing_type == parameter_range_partition::SINGLE){
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
  double midpoint = get_midpoint_of_two_nodes(0,2,&space_range[0],parameter_range_partition::GEOMETRIC);
  int midpoint_idx = std::lower_bound(features.begin()+start_idx,features.begin()+end_idx+1,midpoint) - features.begin();
  // NOTE: Ideally idpoint_idx is roughly (start_idx + (end_idx-start_idx)/2), but sometimes all of the observations exist in one of the two partitions
  if (features[midpoint_idx] >= midpoint || midpoint_idx==end_idx) --midpoint_idx;// handles corner case, prevents infinite loop
  partition_space(max_num_obs_per_cell,start_idx,midpoint_idx,features,nodes,max_spacing_factor);
  partition_space(max_num_obs_per_cell,midpoint_idx+1,end_idx,features,nodes,max_spacing_factor);
}

int get_interval_index(double val, int num_nodes, const double* _nodes, parameter_range_partition node_spacing){
  // Used for mapping configuration into grid-cell (tensor element, equivalently)
  // NOTE: Should never be called for extrapolation! Currently an assert checks for this. The more graceful behavior would be to return the min/max interval if val exists outside.
  if (num_nodes==1) return 0;//CORNER CASE
/* Don't need these asserts anymore. If val is outside of [_nodes[0],_nodes[num_nodes-1]], then return 0 or num_nodes-1, respectively
  assert(val <= _nodes[num_nodes-1]);
  assert(val >= _nodes[0]);
*/
  assert(_nodes[0] < _nodes[num_nodes-1]);
  if (val >= _nodes[num_nodes-1]){
      return node_spacing==parameter_range_partition::SINGLE ? num_nodes-1 : num_nodes-2;
  }
  if (val <= _nodes[0]) return 0;
  // Binary Search
  // Loop invariant : cell index is in [start,end]
  int start=0;
  int end=std::max(num_nodes-2,start);//NOTE: This works for parameter_range_partition::SINGLE spacing too, because we already checked for the last node
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

int get_node_index(double val, int num_nodes, const double* _nodes, parameter_range_partition node_spacing_type){
  //NOTE: This function is different than the function in python version!
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
      if (val <= get_midpoint_of_two_nodes(mid,num_nodes,_nodes,node_spacing_type)) return mid;
      else return mid+1;
    }
    else if (val < _nodes[mid]) end=mid-1;
    else start = mid+1;
  }
  return start;
}

std::vector<double> generate_nodes(double _min, double _max, int num_grid_pts, parameter_range_partition node_spacing_type, parameter_type param_type){
    if (param_type==parameter_type::NUMERICAL && node_spacing_type==parameter_range_partition::SINGLE) assert(0);// Not valid
    if (param_type==parameter_type::NUMERICAL){
      assert(_max > _min);
      assert(num_grid_pts>1);
    }
    else if (param_type==parameter_type::CATEGORICAL){
      assert(_max >= _min);
      assert(num_grid_pts>0);
    }
    std::vector<double> nodes;
    nodes.reserve(num_grid_pts);
    if (node_spacing_type == parameter_range_partition::UNIFORM){
      double spacing = (_max-_min)*1./(num_grid_pts-1);
      double pos = _min;
      for (int i=0; i<num_grid_pts; i++){
        nodes.push_back(pos);
        pos += spacing;
      }
    }
    else if (node_spacing_type == parameter_range_partition::GEOMETRIC){
        assert(_min>0);
        nodes = generate_nodes(log(_min),log(_max),num_grid_pts,parameter_range_partition::UNIFORM,param_type);
        for (int i=0; i<nodes.size(); i++){
          nodes[i] = exp(nodes[i]);
        }
    } else assert(0);
    // Enforce the boundary points
    nodes[0]=_min;
    nodes[nodes.size()-1]=_max;
    return nodes;
}

tensor_model_fit_info::tensor_model_fit_info() : model_fit_info(){}
tensor_model_fit_info::tensor_model_fit_info(const tensor_model_fit_info& rhs) : model_fit_info(rhs){
  this->num_tensor_elements = rhs.num_tensor_elements;
  this->tensor_density = rhs.tensor_density;
  this->quadrature_error = rhs.quadrature_error;
}
tensor_model_fit_info& tensor_model_fit_info::operator=(const tensor_model_fit_info& rhs){
  this->model_fit_info::operator=(rhs);
  this->num_tensor_elements = rhs.num_tensor_elements;
  this->tensor_density = rhs.tensor_density;
  this->quadrature_error = rhs.quadrature_error;
}
tensor_model_fit_info::~tensor_model_fit_info(){
}

cpr_model_fit_info::cpr_model_fit_info() : tensor_model_fit_info(){}
cpr_model_fit_info::cpr_model_fit_info(const cpr_model_fit_info& rhs) : tensor_model_fit_info(rhs){
  this->loss = rhs.loss;
  this->low_rank_approximation_error = rhs.low_rank_approximation_error;
}
cpr_model_fit_info& cpr_model_fit_info::operator=(const cpr_model_fit_info& rhs){
  this->tensor_model_fit_info::operator=(rhs);
  this->loss = rhs.loss;
  this->low_rank_approximation_error = rhs.low_rank_approximation_error;
}
cpr_model_fit_info::~cpr_model_fit_info(){
}

cprg_model_fit_info::cprg_model_fit_info() : cpr_model_fit_info(){}
cprg_model_fit_info::cprg_model_fit_info(const cprg_model_fit_info& rhs) : cpr_model_fit_info(rhs){
}
cprg_model_fit_info& cprg_model_fit_info::operator=(const cprg_model_fit_info& rhs){
  this->cpr_model_fit_info::operator=(rhs);
}
cprg_model_fit_info::~cprg_model_fit_info(){
}

void cpr_model::init(
  std::vector<int>& cells_info, const std::vector<double> custom_grid_pts,
  int num_configurations, const double* features){
    this->m_is_valid=true;
    auto _hyperparameters = dynamic_cast<cpr_hyperparameter_pack*>(this->hyperparameters);
    auto _parameters = dynamic_cast<cpr_parameter_pack*>(this->parameters);

    _parameters->knot_positions.clear();
    _parameters->knot_index_offsets.clear();
    _parameters->num_partitions_per_dimension.resize(this->order);
    _parameters->num_dimensions = this->order;
    _parameters->knot_index_offsets.resize(this->order);
    _parameters->knot_index_offsets[0]=0;
    for (int i=0; i<this->order; i++){
        if (_hyperparameters->partition_spacing[i]==parameter_range_partition::SINGLE){
          assert(features != nullptr);
          // Valid for parameter types: parameter_type::CATEGORICAL and parameter_type::NUMERICAL
          // Each distinct value is a distinct node
          // cells_info[i] not relevant, not parsed.
          std::vector<double> temp_nodes;
          std::vector<double> features_copy(num_configurations);
          for (int j=0; j<num_configurations; j++){
            features_copy[j] = std::max(features[this->order*j+i],this->param_range_min[i]);
          }
          std::sort(features_copy.begin(),features_copy.end());
          features_copy.erase(std::unique(features_copy.begin(),features_copy.end()),features_copy.end());// remove duplicates
          _parameters->num_partitions_per_dimension[i] = features_copy.size();
          for (auto it : features_copy){
            _parameters->knot_positions.push_back(it);
          }
          if (i<(this->order-1)) _parameters->knot_index_offsets[i+1]=features_copy.size()+_parameters->knot_index_offsets[i];
        }
        else if (_hyperparameters->partition_spacing[i] == parameter_range_partition::UNIFORM || _hyperparameters->partition_spacing[i] == parameter_range_partition::GEOMETRIC){
          assert(this->param_types[i]==parameter_type::NUMERICAL);
          auto temp_nodes = generate_nodes(this->param_range_min[i],this->param_range_max[i],cells_info[i]+1,_hyperparameters->partition_spacing[i],this->param_types[i]);
          _parameters->num_partitions_per_dimension[i] = temp_nodes.size() - 1;
          for (auto it : temp_nodes){
            if (_hyperparameters->partition_spacing[i]==parameter_range_partition::GEOMETRIC) assert(it>0);
            _parameters->knot_positions.push_back(it);
          }
          assert(_parameters->num_partitions_per_dimension[i]>0);
          if (i<(this->order-1)) _parameters->knot_index_offsets[i+1]=temp_nodes.size()+_parameters->knot_index_offsets[i];
        } else if (_hyperparameters->partition_spacing[i] == parameter_range_partition::AUTOMATIC){
            assert(features != nullptr);
            assert(this->param_types[i]==parameter_type::NUMERICAL);
            assert(num_configurations > 0);
            int max_num_distinct_obs_per_cell = cells_info[i];
            std::vector<double> temp_nodes;
            std::vector<double> features_copy(num_configurations);
            for (int j=0; j<num_configurations; j++){
              features_copy[j] = std::max(features[this->order*j+i],this->param_range_min[i]);
            }
            std::sort(features_copy.begin(),features_copy.end());
            features_copy.erase(std::unique(features_copy.begin(),features_copy.end()),features_copy.end());// remove duplicates
            temp_nodes.push_back(this->param_range_min[i]);
            // Invariant: don't add the first knot in a range, and the end-points (as indices into features_copy) are always valid, and duplicate values are not present.
            partition_space(max_num_distinct_obs_per_cell,0,features_copy.size()-1,features_copy,temp_nodes,_hyperparameters->max_partition_spacing_factor);
            if (temp_nodes.size()==1 || temp_nodes[temp_nodes.size()-1] < this->param_range_max[i]){
              temp_nodes.push_back(this->param_range_max[i]);
            }
            assert(temp_nodes.size()>1);
            assert(temp_nodes[temp_nodes.size()-1] == this->param_range_max[i]);
            // Below handles a corner case in which all parameter values are the same.
            if (temp_nodes.size()==2 && temp_nodes[1]==temp_nodes[0]){
              assert(0);
              ++temp_nodes[1];
            }
            assert(temp_nodes[0]==this->param_range_min[i]);
            //assert(temp_nodes.size()>1);
            for (int j=0; j<temp_nodes.size(); j++){ if (j>0) { assert(temp_nodes[j]>temp_nodes[j-1]); } _parameters->knot_positions.push_back(temp_nodes[j]); }
            _parameters->num_partitions_per_dimension[i] = temp_nodes.size()-1;
            if (this->param_types[i]==parameter_type::NUMERICAL) assert(_parameters->num_partitions_per_dimension[i]>0);
            if (i<(this->order-1)) _parameters->knot_index_offsets[i+1]=temp_nodes.size()+_parameters->knot_index_offsets[i];
         } else{
            assert(0);//parameter_range_partition::CUSTOM interval spacing not tested yet.
            //auto temp_nodes = generate_nodes(this->param_range_min[i],this->param_range_max[i],cells_info[i],_hyperparameters->partition_spacing[i],std::vector<double>(this->custom_grid_pts.begin()+start_grid_idx,this->custom_grid_pts.begin()+start_grid_idx+cells_info[i]]));
            auto temp_nodes = generate_nodes(this->param_range_min[i],this->param_range_max[i],cells_info[i],_hyperparameters->partition_spacing[i],this->param_types[i]);
            _parameters->num_partitions_per_dimension[i] = temp_nodes.size();
            for (auto it : temp_nodes) _parameters->knot_positions.push_back(it);
            if (i<(this->order-1)) _parameters->knot_index_offsets[i+1]=temp_nodes.size()+_parameters->knot_index_offsets[i];
        }
        assert(_parameters->num_partitions_per_dimension[i]>0);
    }
    _parameters->num_knots = _parameters->knot_positions.size();

    // Debug Info
    std::cout << "Min/Max mode range: " << std::endl;
    for (int i=0; i<this->order; i++){
      std::cout << this->param_range_min[i] << " " << this->param_range_max[i] << std::endl;
    }
    std::cout << "cells per dim:\n";
    for (int i=0; i<this->order; i++){
      std::cout << cells_info[i] << " ";
    }
    std::cout << std::endl;
    int temp_idx = 0;
    for (int i=0; i<this->order; i++){
      for (int j=0; j<_parameters->num_partitions_per_dimension[i]+(this->param_types[i]==parameter_type::NUMERICAL ? 1 : 0); j++){
        std::cout << _parameters->knot_positions[temp_idx++] << " ";
      }
      std::cout << std::endl;
    }
    for (int i=0; i<this->order; i++) std::cout << _parameters->num_partitions_per_dimension[i] << " ";
    std::cout << std::endl;

    this->Projected_Omegas.clear();
    this->Projected_Omegas.resize(this->order);

}

cpr_model::cpr_model(int nparam, const parameter_type* parameter_types, const hyperparameter_pack* pack) : model(nparam,parameter_types,pack){
  const auto _pack = dynamic_cast<const cpr_hyperparameter_pack*>(pack);
  this->hyperparameters = new cpr_hyperparameter_pack(_pack==nullptr ? nparam : *_pack);
  this->parameters = new cpr_parameter_pack();
  auto _hyperparameters = dynamic_cast<cpr_hyperparameter_pack*>(this->hyperparameters);
  auto _parameters = dynamic_cast<cpr_parameter_pack*>(this->parameters);
  this->order = nparam;
  assert(this->order>0);
  // Inspect partition_spacing to make sure none are parameter_range_partition::AUTOMATIC, because we have not implemented that yet.
  for (int i=0; i<_hyperparameters->nparam; i++) assert(_hyperparameters->partition_spacing[i] != parameter_range_partition::CUSTOM);
  for (int i=0; i<this->nparam; i++){
    if (this->param_types[i]==parameter_type::NUMERICAL) this->numerical_modes.push_back(i);
  }

  this->m_is_valid=false;
}

cpr_model::~cpr_model(){
}

double cpr_model::predict(const double* configuration) const{
  auto _hyperparameters = dynamic_cast<cpr_hyperparameter_pack*>(this->hyperparameters);
  auto _parameters = dynamic_cast<cpr_parameter_pack*>(this->parameters);
      // NOTE: Assumes that len of configuration is equal to this->order
      // NOTE: The below is still valid, we just need to make sure that the training always works, even if one of the modes has a range of 0. This is a separate robustness check.
      const double* const& fm = _parameters->factor_matrix_elements;
      const int& rank = _parameters->cp_rank;
      assert(fm!=nullptr);
      assert(rank>=1);

      // Debug
      int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
      bool do_i_want_to_print_out = false;//world_rank==0 && _parameters->num_partitions_per_dimension[0]>10 && _parameters->num_partitions_per_dimension[1]>10;

      // Check for invalid input
      for (int i=0; i<this->order; i++){
        if (configuration[i]<=0 && this->param_types[i]==parameter_type::NUMERICAL) return MIN_POS_RUNTIME;// default "zero-value"
      }

      std::vector<int> node(this->order,-1);
      std::vector<double> midpoints;
      std::vector<int> intervals;
      std::vector<int> modes_to_interpolate;
      std::vector<double> local_interp_map(this->order,0);
      std::vector<int> decisions(this->order,0);
      for (int j=0; j<this->order; j++){
          // Get the closest node (note that node!=midpoint). Two nodes define the bounding box of a grid-cell. Each grid-cell has a mid-point.
          if (this->param_types[j]==parameter_type::CATEGORICAL){
            // Extrapolation is not possible here
            //node[j] = get_interval_index(configuration[j],_parameters->num_partitions_per_dimension[j],&_parameters->knot_positions[_parameters->knot_index_offsets[j]],_hyperparameters->partition_spacing[j]);
            node[j] = get_node_index(configuration[j],_parameters->num_partitions_per_dimension[j],&_parameters->knot_positions[_parameters->knot_index_offsets[j]],_hyperparameters->partition_spacing[j]);
            continue;
          }
          if (_parameters->num_partitions_per_dimension[j]==1){
            //The reason we treat this as a specal case is because linear interpolation is not valid, as there is just one point. Linear extraplation is also invalid. 
            // Further, extrapolation is just not possible in this case, so we don't even check for it.
            node[j] = 0;
            continue;
          }
          // check if configuration[numerical_modes[j]] is outside of the parameter_nodes on either side
          double leftmost_midpoint = get_midpoint_of_two_nodes(0, _parameters->num_partitions_per_dimension[j]+(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 0 : 1),&_parameters->knot_positions[_parameters->knot_index_offsets[j]], _hyperparameters->partition_spacing[j]);
          assert(!(_parameters->num_partitions_per_dimension[j]-(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 2 : 1)+1==_parameters->num_partitions_per_dimension[j]+(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 0 : 1)));
          double rightmost_midpoint = get_midpoint_of_two_nodes(_parameters->num_partitions_per_dimension[j]-(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 2 : 1), _parameters->num_partitions_per_dimension[j]+(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 0 : 1), &_parameters->knot_positions[_parameters->knot_index_offsets[j]], _hyperparameters->partition_spacing[j]);
          int interval_idx = get_interval_index(configuration[j],_parameters->num_partitions_per_dimension[j]+(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 0 : 1),&_parameters->knot_positions[_parameters->knot_index_offsets[j]],_hyperparameters->partition_spacing[j]);
          if (do_i_want_to_print_out){
            std::cout << "leftmost midpoint: " << leftmost_midpoint << ", rightmost midpoint - " << rightmost_midpoint << std::endl;
          }
          if (configuration[j] < _parameters->knot_positions[_parameters->knot_index_offsets[j]]){
              // extrapolation necessary: we simply use the last midpoint
              node[j]=interval_idx;
          } else if (configuration[j] > _parameters->knot_positions[j==(this->order-1) ? _parameters->knot_positions.size()-1 : _parameters->knot_index_offsets[j+1]-1]){
              // extrapolation necessary: we simply use the first midpoint
              node[j]=interval_idx;
          } else if (configuration[j] < leftmost_midpoint && _hyperparameters->partition_spacing[j]!=parameter_range_partition::SINGLE){
              // extrapolation necessary: inside range of bounding box on left, but left of left-most midpoint
              if (do_i_want_to_print_out) std::cout << "Linear extrapolation left\n";
              decisions[j]=1;
          } else if (configuration[j] > rightmost_midpoint && _hyperparameters->partition_spacing[j]!=parameter_range_partition::SINGLE){
              // extrapolation necessary: inside range of bounding box on right, but right of right-most midpoint
              if (do_i_want_to_print_out) std::cout << "Linear extrapolation right\n";
              decisions[j]=2;
          } else if (_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE && configuration[j]==_parameters->knot_positions[_parameters->knot_index_offsets[j]+interval_idx]){
              // Don't even attempt to interpolate in this case, because the result will be the same. This avoids unecessary work.
              node[j]=interval_idx;
          } else{
              // Safeguard checks: use raw factor matrix elements rather than recontructing the entire element, as we want to fix all other factor matrix values anyways!
              bool is_valid_to_interpolate = true;
              // Must check all values in corresponding FM row, not just first.
              int eleft = std::max(0,interval_idx-1);
              int emid = interval_idx;
              int eright = std::min(interval_idx+1,_parameters->num_partitions_per_dimension[j]-1);
              int fm_offset = 0;
              for (int k=0; k<j; k++) fm_offset += _parameters->num_partitions_per_dimension[k]*rank;
              for (int k=0; k<rank; k++){
                int eleft_idx = fm_offset+eleft*rank+k;
                int emid_idx = fm_offset+emid*rank+k;
                int eright_idx = fm_offset+eright*rank+k;
                if (fm[eleft_idx]*fm[emid_idx]<0) { is_valid_to_interpolate = false; break; }
                if (std::abs(log(std::abs(fm[eleft_idx]/std::abs(fm[emid_idx])))) > _hyperparameters->interpolation_factor_tolerance) { is_valid_to_interpolate = false; break; }
                if (fm[eright_idx]*fm[emid_idx]<0) { is_valid_to_interpolate = false; break; }
                if (std::abs(log(std::abs(fm[eright_idx]/std::abs(fm[emid_idx])))) > _hyperparameters->interpolation_factor_tolerance) { is_valid_to_interpolate = false; break; }
              }
              if (!is_valid_to_interpolate){
                node[j]=interval_idx;
              } else{
                // Feature value configuration[j] exists within mid-points, so linear interpolation of tensor elements (which represent the sample mean execution time of the midpoint) is feasible
                intervals.push_back(interval_idx);
                //NOTE: The actual midpoint isn't obvious.
                assert(intervals[intervals.size()-1]>=0);
                if (_hyperparameters->partition_spacing[j]!=parameter_range_partition::SINGLE) assert(!(intervals[intervals.size()-1]+1==_parameters->num_partitions_per_dimension[j]+(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 0 : 1)));
                midpoints.push_back(get_midpoint_of_two_nodes(intervals[intervals.size()-1], _parameters->num_partitions_per_dimension[j]+(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 0 : 1),&_parameters->knot_positions[_parameters->knot_index_offsets[j]], _hyperparameters->partition_spacing[j]));
                if (do_i_want_to_print_out) std::cout << "Interpolation, interval - " << intervals[intervals.size()-1] << ", midpoint - " << midpoints[midpoints.size()-1] << " " << _parameters->knot_positions[_parameters->knot_index_offsets[j]+intervals[intervals.size()-1]] << "," << _parameters->knot_positions[_parameters->knot_index_offsets[j]+intervals[intervals.size()-1]+1] << std::endl;
                modes_to_interpolate.push_back(j);
                local_interp_map[j] = 1;
                decisions[j]=5;
             }
           }
         }

      std::vector<std::pair<int,int>> element_index_modes_list;
      // Because each tensor element is associated with the mid-point of the corresponding grid-cell (as a tensor product of mid-points of each parameter's range), we also want to get the surrounding nodes
      // calculation of the nodes to use for a (multi)linear interpolation strategy is nontrivial
      for (int j=0; j<modes_to_interpolate.size(); j++){
        // NOTE: This loop iteration exists only if there always exists a midpoint to left and right!
        element_index_modes_list.push_back(std::make_pair(-1,-1));
        if (_hyperparameters->partition_spacing[modes_to_interpolate[j]]==parameter_range_partition::SINGLE){
            element_index_modes_list[j].first = std::min(_parameters->num_partitions_per_dimension[modes_to_interpolate[j]]-2,intervals[j]);
            element_index_modes_list[j].second = element_index_modes_list[j].first+1;
        } else{
          if (configuration[modes_to_interpolate[j]] < midpoints[j]){
            element_index_modes_list[j].first = std::min(_parameters->num_partitions_per_dimension[modes_to_interpolate[j]]-2,intervals[j]-1);
            element_index_modes_list[j].second = element_index_modes_list[j].first+1;
          } else{
            element_index_modes_list[j].first = std::min(intervals[j],_parameters->num_partitions_per_dimension[modes_to_interpolate[j]]-2);
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
            int cell_node_idx = _parameters->knot_index_offsets[modes_to_interpolate[l]];
            if (interp_id_list[l] == 0){
              int order_id = modes_to_interpolate[l];
              int& temp_id = element_index_modes_list[l].first;
              assert(temp_id>=0);
              double left_point = _hyperparameters->partition_spacing[order_id]==parameter_range_partition::SINGLE ? _parameters->knot_positions[_parameters->knot_index_offsets[order_id]+temp_id] : get_midpoint_of_two_nodes(temp_id, _parameters->num_partitions_per_dimension[order_id]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[order_id]], _hyperparameters->partition_spacing[order_id]);
              if (_hyperparameters->partition_spacing[order_id]!=parameter_range_partition::SINGLE) assert(!(temp_id+2==_parameters->num_partitions_per_dimension[order_id]+1));
              double right_point = _hyperparameters->partition_spacing[order_id]==parameter_range_partition::SINGLE ? _parameters->knot_positions[_parameters->knot_index_offsets[order_id]+temp_id+1] : get_midpoint_of_two_nodes(temp_id+1, _parameters->num_partitions_per_dimension[order_id]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[order_id]], _hyperparameters->partition_spacing[order_id]);
              if (do_i_want_to_print_out) std::cout << "Points(0): " << left_point << " " << right_point << std::endl;
              coeff *= std::max(0.,(1-(std::abs(configuration[order_id]-left_point))/(right_point-left_point)));
              assert(coeff >= 0);
            }
            // coeff quantifies how close the current point (configuration) is to the mid-point characterized by interp_id_list
            if (interp_id_list[l] == 1){
              int order_id = modes_to_interpolate[l];
              int& temp_id = element_index_modes_list[l].second;
              assert(temp_id>=1);
              //if (_hyperparameters->partition_spacing[order_id]==parameter_range_partition::SINGLE) assert(!(temp_id==_parameters->num_partitions_per_dimension[order_id]+1));
              double left_point = _hyperparameters->partition_spacing[order_id]==parameter_range_partition::SINGLE ? _parameters->knot_positions[_parameters->knot_index_offsets[order_id]+temp_id-1] : get_midpoint_of_two_nodes(temp_id-1, _parameters->num_partitions_per_dimension[order_id]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[order_id]], _hyperparameters->partition_spacing[order_id]);
              //if (_hyperparameters->partition_spacing[order_id]==parameter_range_partition::SINGLE) assert(!(temp_id+1==_parameters->num_partitions_per_dimension[order_id]+1));
              double right_point = _hyperparameters->partition_spacing[order_id]==parameter_range_partition::SINGLE ? _parameters->knot_positions[_parameters->knot_index_offsets[order_id]+temp_id] : get_midpoint_of_two_nodes(temp_id, _parameters->num_partitions_per_dimension[order_id]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[order_id]], _hyperparameters->partition_spacing[order_id]);
              if (do_i_want_to_print_out) std::cout << "Points(1): " << left_point << " " << right_point << std::endl;
              coeff *= std::max(0.,(1-(std::abs(configuration[order_id]-right_point))/(right_point-left_point)));
              assert(coeff >= 0);
            }
          }
          std::vector<double> factor_row_list;
//          factor_row_list.reserve(rank*this->order);
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
                  assert(node[l]<_parameters->num_partitions_per_dimension[l]);// especially important if _parameters->num_partitions_per_dimension[l]==1
                  factor_row_list.push_back(fm[factor_matrix_offset + node[l]*rank+ll]);
                }
              } else if (decisions[l]==1){
                assert(_hyperparameters->partition_spacing[l]!=parameter_range_partition::SINGLE);
                double left_midpoint = get_midpoint_of_two_nodes(0, _parameters->num_partitions_per_dimension[l]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[l]], _hyperparameters->partition_spacing[l]);
                double right_midpoint = get_midpoint_of_two_nodes(1, _parameters->num_partitions_per_dimension[l]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[l]], _hyperparameters->partition_spacing[l]);
                for (int ll=0; ll<rank; ll++){
                  assert(right_midpoint > left_midpoint);
                  double _slope = (fm[factor_matrix_offset+rank+ll]-fm[factor_matrix_offset+ll])/(right_midpoint - left_midpoint);
                  double _num = configuration[l]-left_midpoint;
                  assert(_num <= 0);
                  factor_row_list.push_back(fm[factor_matrix_offset+ll] + _num*_slope);
                }
              } else if (decisions[l]==2){
                assert(_hyperparameters->partition_spacing[l]!=parameter_range_partition::SINGLE);
                double left_midpoint = get_midpoint_of_two_nodes(_parameters->num_partitions_per_dimension[l]-2, _parameters->num_partitions_per_dimension[l]+1, &_parameters->knot_positions[_parameters->knot_index_offsets[l]], _hyperparameters->partition_spacing[l]);
                double right_midpoint = get_midpoint_of_two_nodes(_parameters->num_partitions_per_dimension[l]-1, _parameters->num_partitions_per_dimension[l]+1, &_parameters->knot_positions[_parameters->knot_index_offsets[l]], _hyperparameters->partition_spacing[l]);
                for (int ll=0; ll<rank; ll++){
                  double _slope = (fm[factor_matrix_offset+(_parameters->num_partitions_per_dimension[l]-1)*rank+ll]-fm[factor_matrix_offset+(_parameters->num_partitions_per_dimension[l]-2)*rank+ll]);
                  assert(right_midpoint > left_midpoint);
                  double _denom = right_midpoint - left_midpoint;
                  double _num = configuration[l]-right_midpoint;
                  assert(_num >= 0);
                  factor_row_list.push_back(fm[factor_matrix_offset+(_parameters->num_partitions_per_dimension[l]-1)*rank+ll] + _num/_denom*_slope);
                }
              }
          }
          factor_matrix_offset += _parameters->num_partitions_per_dimension[l]*rank;
        }
        double t_val = multilinear_product_packed(&factor_row_list[0],factor_row_list.size(),this->order,rank);
        if (_hyperparameters->runtime_transform==runtime_transformation::LOG){
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
}

bool cpr_model::train(int& num_configurations, const double*& configurations, const double*& runtimes, bool save_dataset, model_fit_info* fit_info){
  auto _hyperparameters = dynamic_cast<cpr_hyperparameter_pack*>(this->hyperparameters);
  auto _parameters = dynamic_cast<cpr_parameter_pack*>(this->parameters);
  auto _fit_info = dynamic_cast<cpr_model_fit_info*>(fit_info);
  bool compute_fit_error = _fit_info!=nullptr;
      if (!model::train(num_configurations,configurations,runtimes,save_dataset,fit_info)) return false;
      assert(_hyperparameters != nullptr);

      int world_size_for_training; MPI_Comm_size(_hyperparameters->cm_training,&world_size_for_training);
      assert(world_size_for_training==1);

      int my_rank; MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
      int world_rank = my_rank;

      // Update the interval spacing justs in case it was set incorrectly (as default is geometric)
      // NOTE: The user may have specified parameter_range_partition::GEOMETRIC spacing for a parameter_type::NUMERICAL parameter, yet the range is so small that it is clear that a log-like tranformation
      //    was already applied to the parameter values in the training set (e.g., if process count has been transformed to log(process count). In that case,
      //    the spacing should be reset to parameter_range_partition::UNIFORM and a zero value should be allowed.
      for (int j=0; j<this->order; j++){
        if (this->param_types[j]==parameter_type::NUMERICAL && _hyperparameters->partition_spacing[j]==parameter_range_partition::GEOMETRIC && this->param_range_min[j] <= 0){
          _hyperparameters->partition_spacing[j]=parameter_range_partition::UNIFORM;
        }
      }
      for (int j=0; j<this->order; j++){
        if (this->param_types[j]==parameter_type::NUMERICAL && this->param_range_min[j]==this->param_range_max[j]){
          _hyperparameters->partition_spacing[j]=parameter_range_partition::SINGLE;
        }
      }

      //NOTE: Ideally if we have some mode/parameter with a single distinct observation, we would enforce that its type is parameter_type::CATEGORICAL rather than parameter_type::NUMERICAL. However, this would not allow ourselves to be robust to future input configurations in which the value for that parameter is different.

      // If the max boundary value of the range is simply too small for the interval spacing of parameter_type::NUMERICAL parameter to be parameter_range_partition::GEOMETRIC, then reset to parameter_range_partition::UNIFORM.
      // TODO:  In this case, it might also be useful to reset to parameter_range_partition::AUTOMATIC so as to leverage the fact that many values in the (small) range might be unobserved.
      for (int j=0; j<this->order; j++){
        if ((this->param_types[j]==parameter_type::NUMERICAL) && (this->param_range_max[j]<NUMERICAL_PARAM_MIN_OBS_RANGE)){
          _hyperparameters->partition_spacing[j]=parameter_range_partition::SINGLE;
          _hyperparameters->partition_info[j]=this->param_range_max[j]-this->param_range_min[j]+1;
        }
      }
      //NOTE: There is also a corner case in which the parameter type is parameter_type::NUMERICAL and the interval spacing is parameter_range_partition::GEOMETRIC, yet the observed feature values are large with a very small range.
      //      We keep parameter_range_partition::GEOMETRIC spacing, noting that the parameter_range_partition::GEOMETRIC spacing is very similar to parameter_range_partition::UNIFORM for this use case.
      std::vector<int> local_cells_info(this->order);
      for (int j=0; j<this->order; j++){
        //NOTE: Below handles a corner case
        local_cells_info[j] = (_hyperparameters->partition_spacing[j]==parameter_range_partition::AUTOMATIC ? _hyperparameters->partition_info[j] : static_cast<int>(std::min(static_cast<double>(_hyperparameters->partition_info[j]),this->param_range_max[j]-this->param_range_min[j]+1)));
      }
      this->init(local_cells_info,{},num_configurations,configurations);

      // Use dictionaries to save the sizes of Omega_i
      assert(this->Projected_Omegas.size() == this->order);
      for (int i=0; i<this->order; i++){
        this->Projected_Omegas[i].clear();
        assert(this->Projected_Omegas[i].size() == 0);
        for (int j=0; j<_parameters->num_partitions_per_dimension[i]; j++){
          this->Projected_Omegas[i].push_back(0);
        }
        assert(this->Projected_Omegas[i].size() == _parameters->num_partitions_per_dimension[i]);
      }
      uint64_t ntensor_elements = 1;
      for (int i=0; i<_parameters->num_partitions_per_dimension.size(); i++){
          ntensor_elements *= _parameters->num_partitions_per_dimension[i];
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
            assert(configurations[i*this->order+j] >= _parameters->knot_positions[_parameters->knot_index_offsets[j]]);
            assert(configurations[i*this->order+j] <= _parameters->knot_positions[_parameters->knot_index_offsets[j]+_parameters->num_partitions_per_dimension[j]-((this->param_types[j]==parameter_type::NUMERICAL && _hyperparameters->partition_spacing[j]!=parameter_range_partition::SINGLE) ? 0 : 1)]);
            if (this->param_types[j]!= parameter_type::NUMERICAL){
              element_key.push_back(get_node_index(configurations[i*this->order+j],_parameters->num_partitions_per_dimension[j],&_parameters->knot_positions[_parameters->knot_index_offsets[j]],_hyperparameters->partition_spacing[j]));
            } else{
              assert(_parameters->num_partitions_per_dimension[j]>0);
              element_key.push_back(get_interval_index(configurations[i*this->order+j],_parameters->num_partitions_per_dimension[j]+(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 0 : 1),&_parameters->knot_positions[_parameters->knot_index_offsets[j]],_hyperparameters->partition_spacing[j]));
              if (_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE){
                assert(_parameters->knot_positions[_parameters->knot_index_offsets[j]+element_key[j]] == configurations[i*this->order+j]);
              } else{
                assert(_parameters->knot_positions[_parameters->knot_index_offsets[j]+element_key[j]] <= configurations[i*this->order+j]);
                assert(_parameters->knot_positions[_parameters->knot_index_offsets[j]+element_key[j]+1] >= configurations[i*this->order+j]);
              }
            }
            assert(element_key[j] >= 0 && element_key[j] < _parameters->num_partitions_per_dimension[j]);
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
            scale *= _parameters->num_partitions_per_dimension[i]; 
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
      if (compute_fit_error){
        _fit_info->tensor_density = density;
        _fit_info->num_tensor_elements = ntensor_elements;
      }
      for (int i=0; i<this->order; i++){
        for (int j=0; j<this->Projected_Omegas[i].size(); j++){
          std::cout << this->Projected_Omegas[i][j] << " ";
        }
        std::cout << "\n";
      }


      std::vector<double> ones(training_nodes.size(),1.);
      std::vector<int> tensor_mode_types(this->order,NS);
      int64_t num_nnz_elems,num_nnz_elems2,num_nnz_elems3;
      int64_t* ptr_to_indices,*ptr_to_indices2,*ptr_to_indices3;
      double* ptr_to_data,*ptr_to_data2,*ptr_to_data3;
      // NOTE: Not interested in doing any distributed-memory tensor contractions here, as this must proceed online.
      CTF::World dw(_hyperparameters->cm_training);
      CTF::Tensor<> omega(this->order, true, &_parameters->num_partitions_per_dimension[0], &tensor_mode_types[0], dw);
      CTF::Tensor<> Tsparse(this->order, true, &_parameters->num_partitions_per_dimension[0], &tensor_mode_types[0], dw);
      omega.write(training_nodes.size(),&training_nodes[0],&ones[0]);
      Tsparse.write(training_nodes.size(),&training_nodes[0],&training_node_data[0]);

      std::vector<CTF::Tensor<>*> FM1;// Must be Tensor, not Matrix due to constraints of multilinear interface
      std::vector<int> fm_mode_types(2,NS);
      for (int i=0; i<this->order; i++){
          std::vector<int> fm_mode_lengths = {_hyperparameters->cp_rank,_parameters->num_partitions_per_dimension[i]};
          FM1.emplace_back(new CTF::Tensor<>(2,&fm_mode_lengths[0],&fm_mode_types[0],dw));
       }
       // Optimize model

       // For interpolation, we first minimize mean squared error using log-transformed data
       auto _T_ = Tsparse;
       if (_hyperparameters->runtime_transform == runtime_transformation::LOG){
         CTF::Sparse_log(&_T_); 
       }

       int num_re_inits = 0;
      if (_hyperparameters->loss == loss_function::MSE){
        while (num_re_inits < _hyperparameters->max_num_re_inits){
         for (int i=0; i<this->order; i++) init_factor_matrix(FM1[i],_hyperparameters->loss);
         double loss_value = cpd_als(&dw,&_T_,&omega,FM1,_hyperparameters->regularization,_hyperparameters->optimization_convergence_tolerance,_hyperparameters->max_num_optimization_sweeps);
         if (compute_fit_error) _fit_info->loss = loss_value;
         if (loss_value <= _hyperparameters->optimization_convergence_tolerance_for_re_init) break;
         num_re_inits++;
        }
      } else if (_hyperparameters->loss == loss_function::MLOGQ2){
         while (num_re_inits < _hyperparameters->max_num_re_inits){
           for (int i=0; i<this->order; i++) init_factor_matrix(FM1[i],_hyperparameters->loss);
           double loss_value = cpd_amn(&dw,&Tsparse,&omega,FM1,_hyperparameters->regularization,\
             _hyperparameters->optimization_convergence_tolerance,_hyperparameters->max_num_optimization_sweeps, _hyperparameters->factor_matrix_optimization_convergence_tolerance,\
             _hyperparameters->factor_matrix_optimization_max_num_iterations, _hyperparameters->optimization_barrier_start,_hyperparameters->optimization_barrier_stop,_hyperparameters->optimization_barrier_reduction_factor);
           if (compute_fit_error) _fit_info->loss = loss_value;
           if (loss_value <= _hyperparameters->optimization_convergence_tolerance_for_re_init) break;
           num_re_inits++;
         }
      } else assert(0);

      //NOTE: Factor matrices are written as: tensor mode grows slowest, row of FM grows fastest

      int total_elements_across_all_factor_matrices = 0;
      for (int i=0; i<this->order; i++){
        total_elements_across_all_factor_matrices += _parameters->num_partitions_per_dimension[i];
      }
      total_elements_across_all_factor_matrices *= _hyperparameters->cp_rank;
      double* temporary_buffer = new double[total_elements_across_all_factor_matrices];
      int offset = 0;
      for (int i=0; i<this->order; i++){
        FM1[i]->get_local_data(&num_nnz_elems,&ptr_to_indices,&ptr_to_data);
        assert(_parameters->num_partitions_per_dimension[i]*_hyperparameters->cp_rank == num_nnz_elems);
        std::memcpy(temporary_buffer+offset,ptr_to_data,num_nnz_elems*sizeof(double));
        offset += num_nnz_elems;
        delete[] ptr_to_data;
        delete[] ptr_to_indices;
      }
      if (_parameters->factor_matrix_elements!=nullptr) delete[] _parameters->factor_matrix_elements;
      _parameters->factor_matrix_elements = temporary_buffer;
      _parameters->num_factor_matrix_elements = total_elements_across_all_factor_matrices;
      _parameters->cp_rank = _hyperparameters->cp_rank;
      for (int i=0; i<this->order; i++){
        delete FM1[i];
      }

      this->m_is_valid=true;
      if (compute_fit_error){
        double agg_relative_error=0;
        double agg_quadrature_error=0;
        double agg_low_rank_error=0;

        bool do_i_want_to_print_out = false;//world_rank==0 && _parameters->num_partitions_per_dimension[0]>10 && _parameters->num_partitions_per_dimension[1]>10;
        if (do_i_want_to_print_out){
          std::cout << "Start of training set prediction\n";
          std::cout << _parameters->num_partitions_per_dimension[0] << " " << _parameters->num_partitions_per_dimension[1] << "\n";
          std::cout << _parameters->knot_index_offsets[0] << " " << _parameters->knot_index_offsets[1] << "\n";
          for (int i=0; i<_parameters->knot_positions.size(); i++){
            std::cout << _parameters->knot_positions[i] << " ";
          }
          std::cout << "\n";
        }
/*
        // debug: print out factor matrices
        int read = 0;
        for (int i=0; i<3; i++){
          for (int j=0; j<_parameters->num_partitions_per_dimension[i]; j++){
            for (int k=0; k<_parameters->cp_rank; k++){
              std::cout << _parameters->factor_matrix_elements[read++] << " ";
            } std::cout << "\n";
          } std::cout << "\n\n";
        }
*/
        for (int i=0; i<num_configurations; i++){
          if (do_i_want_to_print_out) std::cout << "Predicting from training set (" << configurations[i*this->order] << "," << configurations[i*this->order+1] << std::endl;
          double runtime_estimate = this->cpr_model::predict(&configurations[i*this->order]);
          double rel_err = runtime_estimate<=0 ? 1 : std::abs(log(runtime_estimate/runtimes[i]));
          if (do_i_want_to_print_out) std::cout << "Runtime estimate - " << runtime_estimate << ", true runtime - " << runtimes[i] << ", rel_err - " << rel_err << std::endl;
          std::vector<int> element_key;
          std::vector<double> element_key_configuration;
          for (int j=0; j<this->order; j++){
            assert(configurations[i*this->order+j] <= _parameters->knot_positions[_parameters->knot_index_offsets[j]+_parameters->num_partitions_per_dimension[j]-(this->param_types[j]==parameter_type::NUMERICAL && _hyperparameters->partition_spacing[j]!=parameter_range_partition::SINGLE ? 0 : 1)]);
            if (this->param_types[j]!= parameter_type::NUMERICAL){
              element_key.push_back(get_node_index(configurations[i*this->order+j],_parameters->num_partitions_per_dimension[j],&_parameters->knot_positions[_parameters->knot_index_offsets[j]],_hyperparameters->partition_spacing[j]));
            } else{
              assert(_parameters->num_partitions_per_dimension[j]>0);
              element_key.push_back(get_interval_index(configurations[i*this->order+j],_parameters->num_partitions_per_dimension[j]+(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 0 : 1),&_parameters->knot_positions[_parameters->knot_index_offsets[j]],_hyperparameters->partition_spacing[j]));
              if (_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE){
                assert(_parameters->knot_positions[_parameters->knot_index_offsets[j]+element_key[j]] == configurations[i*this->order+j]);
              } else{
                assert(_parameters->knot_positions[_parameters->knot_index_offsets[j]+element_key[j]] <= configurations[i*this->order+j]);
                assert(_parameters->knot_positions[_parameters->knot_index_offsets[j]+element_key[j]+1] >= configurations[i*this->order+j]);
              }
            }
            assert(element_key[j] >= 0 && element_key[j] < _parameters->num_partitions_per_dimension[j]);
            if (this->param_types[j]==parameter_type::CATEGORICAL){
              element_key_configuration.push_back(_parameters->knot_positions[_parameters->knot_index_offsets[j]+element_key[j]]);
            } else{
              //if (_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE) assert(!(element_key[j]+1==_parameters->num_partitions_per_dimension[j]+1));
              element_key_configuration.push_back(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? _parameters->knot_positions[_parameters->knot_index_offsets[j]+element_key[j]] : get_midpoint_of_two_nodes(element_key[j],_parameters->num_partitions_per_dimension[j]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[j]],_hyperparameters->partition_spacing[j]));
            }
          }
          assert(node_data_dict.find(element_key) != node_data_dict.end());
          assert(node_count_dict.find(element_key) != node_count_dict.end());
          double tensor_elem_sample_mean = node_data_dict[element_key] / node_count_dict[element_key];// use this rather than training_nodes because I don't care about calculating tensor_elem as one integer
          double quad_err = std::abs(log(tensor_elem_sample_mean/runtimes[i]));
          double tensor_elem_prediction = multilinear_product(&_parameters->factor_matrix_elements[0],&element_key[0],&_parameters->num_partitions_per_dimension[0],this->order,_parameters->cp_rank);
          if (_hyperparameters->runtime_transform == runtime_transformation::LOG) tensor_elem_prediction = exp(tensor_elem_prediction);
          double low_rank_err1 = std::abs(log(tensor_elem_prediction/tensor_elem_sample_mean));
          double tensor_elem_prediction2 = this->cpr_model::predict(&element_key_configuration[0]);
          double low_rank_err2 = std::abs(log(tensor_elem_prediction2/tensor_elem_sample_mean));
/*
          std::cout << tensor_elem_prediction2 << " " << tensor_elem_prediction << std::endl;
          if (std::abs(tensor_elem_prediction2-tensor_elem_prediction)>1e-6){
            for (int j=0; j<_parameters->knot_positions.size(); j++) std::cout << _parameters->knot_positions[j] << " ";
            std::cout << "\n" << _parameters->knot_index_offsets[0] << " " << _parameters->knot_index_offsets[1] << " " << _parameters->knot_index_offsets[2] << "\n";
            std::cout << configurations[i*this->order] << " " << configurations[i*this->order+1] << " " << configurations[i*this->order+2] << " " << configurations[i*this->order+3] << " " << configurations[i*this->order+4] << " " << configurations[i*this->order+5] << " " << configurations[i*this->order+6] << " " << configurations[i*this->order+7] << " " << element_key[0] << " " << element_key[1] << " " << element_key[2] << " " << element_key[3] << " " << element_key[4] << " " << element_key[5] << " " << element_key[6] << " " << element_key[7] << " " << element_key_configuration[0] << " " << element_key_configuration[1] << " " << element_key_configuration[2] << " " << element_key_configuration[3] << " " << element_key_configuration[4] << " " << element_key_configuration[5] << " " << element_key_configuration[6] << " " << element_key_configuration[7] << " " << tensor_elem_prediction << " " << tensor_elem_prediction2 << std::endl;
         }
*/
//          assert(std::abs(tensor_elem_prediction2-tensor_elem_prediction)<=1e-6); 
          if (do_i_want_to_print_out && quad_err <= .1 && low_rank_err1 <= .1 && rel_err >= .5){
            // get nearby predictions too (just for CTF)
            std::vector<int> element_key_left = {element_key[0],std::max(0,element_key[1]-1)};
            std::vector<int> element_key_right = {element_key[0],std::min(_parameters->num_partitions_per_dimension[1],element_key[1]+1)};
            std::vector<double> element_key_config_left = {element_key_configuration[0],get_midpoint_of_two_nodes(element_key_left[1],_parameters->num_partitions_per_dimension[1]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[1]],_hyperparameters->partition_spacing[1])};
            std::vector<double> element_key_config_right = {element_key_configuration[0],get_midpoint_of_two_nodes(element_key_right[1],_parameters->num_partitions_per_dimension[1]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[1]],_hyperparameters->partition_spacing[1])};
            double tensor_elem_left_prediction3 = this->cpr_model::predict(&element_key_config_left[0]);
            double tensor_elem_right_prediction3 = this->cpr_model::predict(&element_key_config_right[0]);
            std::cout << "(" << configurations[i*this->order] << "," << configurations[i*this->order+1] << "), Node: (" << element_key[0] << "," << element_key[1] << "), Element config - (" << element_key_configuration[0] << "," << element_key_configuration[1] << "), Left: - " << tensor_elem_left_prediction3 << ", Right: " << tensor_elem_right_prediction3 << ", " << this->Projected_Omegas[1][element_key_left[1]] << "," << this->Projected_Omegas[1][element_key[1]] << "," << this->Projected_Omegas[1][element_key_right[1]] << "  , True Runtime: " << runtimes[i] << ", Runtime Preditions: (" << tensor_elem_sample_mean << " , " << tensor_elem_prediction << " , " << tensor_elem_prediction2 << " , " << runtime_estimate << "), Errors: (" << quad_err << " , " << low_rank_err1 << " , " << low_rank_err2 << " , " << rel_err << ")\n";
          }
          //NOTE: I could also check correlation between high relative error and high quadrature error
          //std::cout << rel_err << " " << agg_relative_error << " " << runtime_estimate << " " << runtimes[i] << " " << configurations[i*this->order] << " " << configurations[i*this->order+1] << " " << configurations[i*this->order+2] << std::endl;
          agg_relative_error += rel_err;
          agg_quadrature_error += quad_err;
        }
        if (do_i_want_to_print_out){
          std::cout << "\n\n\n\n\n";
        }
        for (auto& it : node_data_dict){
          auto& element_key = it.first;
          double tensor_elem_sample_mean = node_data_dict[element_key] / node_count_dict[element_key];// use this rather than training_nodes because I don't care about calculating tensor_elem as one integer
          double tensor_elem_prediction = multilinear_product(&_parameters->factor_matrix_elements[0],&element_key[0],&_parameters->num_partitions_per_dimension[0],this->order,_parameters->cp_rank);
          if (_hyperparameters->runtime_transform == runtime_transformation::LOG) tensor_elem_prediction = exp(tensor_elem_prediction);
          double low_rank_err = std::abs(log(tensor_elem_prediction/tensor_elem_sample_mean));
//          std::cout << low_rank_err << " " << tensor_elem_sample_mean << " " << tensor_elem_prediction << std::endl;
          //NOTE: I could also check correlation between high relative error and high quadrature error
          agg_low_rank_error += low_rank_err;
        }
        agg_relative_error /= num_configurations;
        agg_quadrature_error /= num_configurations;
        agg_low_rank_error /= node_data_dict.size();
        _fit_info->training_error = agg_relative_error;
        _fit_info->quadrature_error = agg_quadrature_error;
        _fit_info->low_rank_approximation_error = agg_low_rank_error;
      }
      if (this->allocated_data){
        delete[] configurations;
        delete[] runtimes;
      }
      return true;
}

void cpr_model::write_to_file(const char* file_path) const{
  std::ofstream model_file_ptr;
  // Will overwrite anything in existing file
  model_file_ptr.open(file_path,std::ios_base::out);
  if(model_file_ptr.fail()) return;
  this->write_to_file(model_file_ptr);
  model_file_ptr.close();
}
void cpr_model::read_from_file(const char* file_path){
  std::ifstream model_file_ptr;
  // Will overwrite anything in existing file
  model_file_ptr.open(file_path,std::ios_base::in);
  if(model_file_ptr.fail()) return;
  this->read_from_file(model_file_ptr);
  model_file_ptr.close();
}

void cpr_model::write_to_file(std::ofstream& file) const{
  // Nothing local to cpr_model to write
  this->model::write_to_file(file);
}

void cpr_model::read_from_file(std::ifstream& file){
  // Nothing local to cpr_model to read
  this->m_is_valid = true;
  // order, numerical_modes, categorical_modes should all be set correctly upon invocation of constructor.
  // Projected_Omegas does not need to be read in, as it is not used in predict(..) and is reset upon invocation of train(..)
  this->model::read_from_file(file);
}

void cpr_model::get_hyperparameters(hyperparameter_pack& pack) const {
  this->hyperparameters->get(dynamic_cast<cpr_hyperparameter_pack&>(pack));
}

void cpr_model::set_hyperparameters(const hyperparameter_pack& pack){
  this->hyperparameters->set(dynamic_cast<const cpr_hyperparameter_pack&>(pack));
}

void cpr_model::get_parameters(parameter_pack& pack) const{
  this->parameters->get(dynamic_cast<cpr_parameter_pack&>(pack));
}

void cpr_model::set_parameters(const parameter_pack& pack){
  this->parameters->set(dynamic_cast<const cpr_parameter_pack&>(pack));
}

cprg_model::cprg_model(int nparam, const parameter_type* parameter_types, const hyperparameter_pack* pack) : cpr_model(nparam,parameter_types,pack){
  const auto _pack = dynamic_cast<const cprg_hyperparameter_pack*>(pack);
  this->hyperparameters = new cprg_hyperparameter_pack(_pack==nullptr ? nparam : *_pack);
  this->parameters = new cprg_parameter_pack();
}

cprg_model::~cprg_model(){
}

double cprg_model::predict(const double* configuration) const{
  auto _hyperparameters = dynamic_cast<cprg_hyperparameter_pack*>(this->hyperparameters);
  auto _parameters = dynamic_cast<cprg_parameter_pack*>(this->parameters);
      // NOTE: Assumes that len of configuration is equal to this->order
      // NOTE: The below is still valid, we just need to make sure that the training always works, even if one of the modes has a range of 0. This is a separate robustness check.
      const double* const& fm = _parameters->factor_matrix_elements;
      const double* const& fmesvd = _parameters->global_models;
      const int& rank = _parameters->cp_rank;
      assert(fm!=nullptr);
      assert(fmesvd!=nullptr);
      assert(_hyperparameters != nullptr);
      assert(rank>=1);

      // Debug
      int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
      bool do_i_want_to_print_out = false;//world_rank==0 && _parameters->num_partitions_per_dimension[0]>10 && _parameters->num_partitions_per_dimension[1]>10;

      // Check for invalid input
      for (int i=0; i<this->order; i++){
        if (configuration[i]<=0 && this->param_types[i]==parameter_type::NUMERICAL) return MIN_POS_RUNTIME;// default "zero-value"
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
          if (this->param_types[j]==parameter_type::CATEGORICAL){
            // Extrapolation is not possible here
            node[j] = get_interval_index(configuration[j],_parameters->num_partitions_per_dimension[j],&_parameters->knot_positions[_parameters->knot_index_offsets[j]],_hyperparameters->partition_spacing[j]);
            continue;
          }
          if (_parameters->num_partitions_per_dimension[j]==1){
            //The reason we treat this as a specal case is because linear interpolation is not valid, as there is just one point. Linear extraplation is also invalid. 
            // Further, extrapolation is just not possible in this case, so we don't even check for it.
            node[j] = 0;
            continue;
          }
          // check if configuration[numerical_modes[j]] is outside of the parameter_nodes on either side
          double leftmost_midpoint = get_midpoint_of_two_nodes(0, _parameters->num_partitions_per_dimension[j]+(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 0 : 1),&_parameters->knot_positions[_parameters->knot_index_offsets[j]], _hyperparameters->partition_spacing[j]);
          assert(!(_parameters->num_partitions_per_dimension[j]-(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 2 : 1)+1==_parameters->num_partitions_per_dimension[j]+(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 0 : 1)));
          double rightmost_midpoint = get_midpoint_of_two_nodes(_parameters->num_partitions_per_dimension[j]-(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 2 : 1), _parameters->num_partitions_per_dimension[j]+(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 0 : 1), &_parameters->knot_positions[_parameters->knot_index_offsets[j]], _hyperparameters->partition_spacing[j]);
          int interval_idx = get_interval_index(configuration[j],_parameters->num_partitions_per_dimension[j]+(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 0 : 1),&_parameters->knot_positions[_parameters->knot_index_offsets[j]],_hyperparameters->partition_spacing[j]);
          if (do_i_want_to_print_out){
            std::cout << "leftmost midpoint: " << leftmost_midpoint << ", rightmost midpoint - " << rightmost_midpoint << std::endl;
          }
          if (configuration[j] < _parameters->knot_positions[_parameters->knot_index_offsets[j]]){
              // extrapolation necessary: outside range of bounding box on left
              decisions[j]=3;
              is_extrapolation=true;
          } else if (configuration[j] > _parameters->knot_positions[j==(this->order-1) ? _parameters->knot_positions.size()-1 : _parameters->knot_index_offsets[j+1]-1]){
              // extrapolation necessary: outside range of bounding box on right
              decisions[j]=4;
              is_extrapolation=true;
          } else if (configuration[j] < leftmost_midpoint && _hyperparameters->partition_spacing[j]!=parameter_range_partition::SINGLE){
              // extrapolation necessary: inside range of bounding box on left, but left of left-most midpoint
              if (do_i_want_to_print_out) std::cout << "Linear extrapolation left\n";
              decisions[j]=1;
          } else if (configuration[j] > rightmost_midpoint && _hyperparameters->partition_spacing[j]!=parameter_range_partition::SINGLE){
              // extrapolation necessary: inside range of bounding box on right, but right of right-most midpoint
              if (do_i_want_to_print_out) std::cout << "Linear extrapolation right\n";
              decisions[j]=2;
          } else if (_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE && configuration[j]==_parameters->knot_positions[_parameters->knot_index_offsets[j]+interval_idx]){
              // Don't even attempt to interpolate in this case, because the result will be the same. This avoids unecessary work.
              node[j]=interval_idx;
          } else{
              // Safeguard checks: use raw factor matrix elements rather than recontructing the entire element, as we want to fix all other factor matrix values anyways!
              bool is_valid_to_interpolate = true;
              // Must check all values in corresponding FM row, not just first.
              int eleft = std::max(0,interval_idx-1);
              int emid = interval_idx;
              int eright = std::min(interval_idx+1,_parameters->num_partitions_per_dimension[j]-1);
              int fm_offset = 0;
              for (int k=0; k<j; k++) fm_offset += _parameters->num_partitions_per_dimension[k]*rank;
              for (int k=0; k<rank; k++){
                int eleft_idx = fm_offset+eleft*rank+k;
                int emid_idx = fm_offset+emid*rank+k;
                int eright_idx = fm_offset+eright*rank+k;
                if (fm[eleft_idx]*fm[emid_idx]<0) { is_valid_to_interpolate = false; break; }
                if (std::abs(log(std::abs(fm[eleft_idx]/std::abs(fm[emid_idx])))) > _hyperparameters->interpolation_factor_tolerance) { is_valid_to_interpolate = false; break; }
                if (fm[eright_idx]*fm[emid_idx]<0) { is_valid_to_interpolate = false; break; }
                if (std::abs(log(std::abs(fm[eright_idx]/std::abs(fm[emid_idx])))) > _hyperparameters->interpolation_factor_tolerance) { is_valid_to_interpolate = false; break; }
              }
              if (!is_valid_to_interpolate){
                node[j]=interval_idx;
              } else{
                // Feature value configuration[j] exists within mid-points, so linear interpolation of tensor elements (which represent the sample mean execution time of the midpoint) is feasible
                intervals.push_back(interval_idx);
                //NOTE: The actual midpoint isn't obvious.
                assert(intervals[intervals.size()-1]>=0);
                if (_hyperparameters->partition_spacing[j]!=parameter_range_partition::SINGLE) assert(!(intervals[intervals.size()-1]+1==_parameters->num_partitions_per_dimension[j]+(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 0 : 1)));
                midpoints.push_back(get_midpoint_of_two_nodes(intervals[intervals.size()-1], _parameters->num_partitions_per_dimension[j]+(_hyperparameters->partition_spacing[j]==parameter_range_partition::SINGLE ? 0 : 1),&_parameters->knot_positions[_parameters->knot_index_offsets[j]], _hyperparameters->partition_spacing[j]));
                if (do_i_want_to_print_out) std::cout << "Interpolation, interval - " << intervals[intervals.size()-1] << ", midpoint - " << midpoints[midpoints.size()-1] << " " << _parameters->knot_positions[_parameters->knot_index_offsets[j]+intervals[intervals.size()-1]] << "," << _parameters->knot_positions[_parameters->knot_index_offsets[j]+intervals[intervals.size()-1]+1] << std::endl;
                modes_to_interpolate.push_back(j);
                local_interp_map[j] = 1;
                decisions[j]=5;
             }
           }
         }

      std::vector<std::pair<int,int>> element_index_modes_list;
      // Because each tensor element is associated with the mid-point of the corresponding grid-cell (as a tensor product of mid-points of each parameter's range), we also want to get the surrounding nodes
      // calculation of the nodes to use for a (multi)linear interpolation strategy is nontrivial
      for (int j=0; j<modes_to_interpolate.size(); j++){
        // NOTE: This loop iteration exists only if there always exists a midpoint to left and right!
        element_index_modes_list.push_back(std::make_pair(-1,-1));
        if (_hyperparameters->partition_spacing[modes_to_interpolate[j]]==parameter_range_partition::SINGLE){
          element_index_modes_list[j].first = std::min(_parameters->num_partitions_per_dimension[modes_to_interpolate[j]]-2,intervals[j]);
          element_index_modes_list[j].second = element_index_modes_list[j].first+1;
        } else{
          if (configuration[modes_to_interpolate[j]] < midpoints[j]){
            element_index_modes_list[j].first = std::min(_parameters->num_partitions_per_dimension[modes_to_interpolate[j]]-2,intervals[j]-1);
            element_index_modes_list[j].second = element_index_modes_list[j].first+1;
          } else{
            element_index_modes_list[j].first = std::min(intervals[j],_parameters->num_partitions_per_dimension[modes_to_interpolate[j]]-2);
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
            int cell_node_idx = _parameters->knot_index_offsets[modes_to_interpolate[l]];
            if (interp_id_list[l] == 0){
              int order_id = modes_to_interpolate[l];
              int& temp_id = element_index_modes_list[l].first;
              assert(temp_id>=0);
              double left_point = _hyperparameters->partition_spacing[order_id]==parameter_range_partition::SINGLE ? _parameters->knot_positions[_parameters->knot_index_offsets[order_id]+temp_id] : get_midpoint_of_two_nodes(temp_id, _parameters->num_partitions_per_dimension[order_id]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[order_id]], _hyperparameters->partition_spacing[order_id]);
              if (_hyperparameters->partition_spacing[order_id]!=parameter_range_partition::SINGLE) assert(!(temp_id+2==_parameters->num_partitions_per_dimension[order_id]+1));
              double right_point = _hyperparameters->partition_spacing[order_id]==parameter_range_partition::SINGLE ? _parameters->knot_positions[_parameters->knot_index_offsets[order_id]+temp_id+1] : get_midpoint_of_two_nodes(temp_id+1, _parameters->num_partitions_per_dimension[order_id]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[order_id]], _hyperparameters->partition_spacing[order_id]);
              if (do_i_want_to_print_out) std::cout << "Points(0): " << left_point << " " << right_point << std::endl;
              coeff *= std::max(0.,(1-(std::abs(configuration[order_id]-left_point))/(right_point-left_point)));
              assert(coeff >= 0);
            }
            // coeff quantifies how close the current point (configuration) is to the mid-point characterized by interp_id_list
            if (interp_id_list[l] == 1){
              int order_id = modes_to_interpolate[l];
              int& temp_id = element_index_modes_list[l].second;
              assert(temp_id>=1);
              //if (_hyperparameters->partition_spacing[order_id]==parameter_range_partition::SINGLE) assert(!(temp_id==_parameters->num_partitions_per_dimension[order_id]+1));
              double left_point = _hyperparameters->partition_spacing[order_id]==parameter_range_partition::SINGLE ? _parameters->knot_positions[_parameters->knot_index_offsets[order_id]+temp_id-1] : get_midpoint_of_two_nodes(temp_id-1, _parameters->num_partitions_per_dimension[order_id]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[order_id]], _hyperparameters->partition_spacing[order_id]);
              //if (_hyperparameters->partition_spacing[order_id]==parameter_range_partition::SINGLE) assert(!(temp_id+1==_parameters->num_partitions_per_dimension[order_id]+1));
              double right_point = _hyperparameters->partition_spacing[order_id]==parameter_range_partition::SINGLE ? _parameters->knot_positions[_parameters->knot_index_offsets[order_id]+temp_id] : get_midpoint_of_two_nodes(temp_id, _parameters->num_partitions_per_dimension[order_id]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[order_id]], _hyperparameters->partition_spacing[order_id]);
              if (do_i_want_to_print_out) std::cout << "Points(1): " << left_point << " " << right_point << std::endl;
              coeff *= std::max(0.,(1-(std::abs(configuration[order_id]-right_point))/(right_point-left_point)));
              assert(coeff >= 0);
            }
          }
          std::vector<double> factor_row_list;
//          factor_row_list.reserve(rank*this->order);
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
                  assert(node[l]<_parameters->num_partitions_per_dimension[l]);// especially important if _parameters->num_partitions_per_dimension[l]==1
                  factor_row_list.push_back(fm[factor_matrix_offset + node[l]*rank+ll]);
                }
              } else if (decisions[l]==1){
                assert(_hyperparameters->partition_spacing[l]!=parameter_range_partition::SINGLE);
                double left_midpoint = get_midpoint_of_two_nodes(0, _parameters->num_partitions_per_dimension[l]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[l]], _hyperparameters->partition_spacing[l]);
                double right_midpoint = get_midpoint_of_two_nodes(1, _parameters->num_partitions_per_dimension[l]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[l]], _hyperparameters->partition_spacing[l]);
                for (int ll=0; ll<rank; ll++){
                  assert(right_midpoint > left_midpoint);
                  double _slope = (fm[factor_matrix_offset+rank+ll]-fm[factor_matrix_offset+ll])/(right_midpoint - left_midpoint);
                  double _num = configuration[l]-left_midpoint;
                  assert(_num <= 0);
                  factor_row_list.push_back(fm[factor_matrix_offset+ll] + _num*_slope);
                }
              } else if (decisions[l]==2){
                assert(_hyperparameters->partition_spacing[l]!=parameter_range_partition::SINGLE);
                double left_midpoint = get_midpoint_of_two_nodes(_parameters->num_partitions_per_dimension[l]-2, _parameters->num_partitions_per_dimension[l]+1, &_parameters->knot_positions[_parameters->knot_index_offsets[l]], _hyperparameters->partition_spacing[l]);
                double right_midpoint = get_midpoint_of_two_nodes(_parameters->num_partitions_per_dimension[l]-1, _parameters->num_partitions_per_dimension[l]+1, &_parameters->knot_positions[_parameters->knot_index_offsets[l]], _hyperparameters->partition_spacing[l]);
                for (int ll=0; ll<rank; ll++){
                  double _slope = (fm[factor_matrix_offset+(_parameters->num_partitions_per_dimension[l]-1)*rank+ll]-fm[factor_matrix_offset+(_parameters->num_partitions_per_dimension[l]-2)*rank+ll]);
                  assert(right_midpoint > left_midpoint);
                  double _denom = right_midpoint - left_midpoint;
                  double _num = configuration[l]-right_midpoint;
                  assert(_num >= 0);
                  factor_row_list.push_back(fm[factor_matrix_offset+(_parameters->num_partitions_per_dimension[l]-1)*rank+ll] + _num/_denom*_slope);
                }
              } else{
                  double ppp = 0;
                  double input_scale=1;
                  for (int kk=0; kk<1+_parameters->spline_degree; kk++){
                    ppp += input_scale*fmesvd[fmesvd_offset+kk];
                    input_scale *= (_hyperparameters->factor_matrix_underlying_position_transformation == parameter_transformation::LOG ? log(configuration[l]) : configuration[l]);
                  }
                  for (int lll=0; lll<rank; lll++){
                    factor_row_list.push_back((_hyperparameters->factor_matrix_element_transformation == runtime_transformation::LOG ? exp(ppp) : ppp)*fmesvd[fmesvd_offset+1+_parameters->spline_degree]*fmesvd[fmesvd_offset+2+_parameters->spline_degree+lll]);
                  }
            }
          }
          factor_matrix_offset += _parameters->num_partitions_per_dimension[l]*rank;
          if (this->param_types[l]==parameter_type::NUMERICAL) fmesvd_offset += (2+_hyperparameters->max_spline_degree+rank);
        }
        double t_val = multilinear_product_packed(&factor_row_list[0],factor_row_list.size(),this->order,rank);
        if (_hyperparameters->runtime_transform==runtime_transformation::LOG){
          t_val = exp(t_val);
        }
        if (do_i_want_to_print_out) std::cout << "Coeff - " << coeff << std::endl;
        model_val += coeff * t_val;
      }
      if (do_i_want_to_print_out){
        std::cout << model_val << " " << decisions[0] << " " << decisions[1] << " " << " " << decisions[2] << " " << configuration[0] << " " << configuration[1] << std::endl;
      }
      //assert(model_val>0);
      return std::max(MIN_POS_RUNTIME,model_val);// Only way MIN_POS_RUNTIME is used is if linear extrapolation (decisions 1 or 2) is used  and it is inaccurate.
}

bool cprg_model::train(int& num_configurations, const double*& configurations, const double*& runtimes, bool save_dataset, model_fit_info* fit_info){
      if (!cpr_model::train(num_configurations,configurations,runtimes,save_dataset,fit_info)) return false;
      auto _hyperparameters = dynamic_cast<cprg_hyperparameter_pack*>(this->hyperparameters);
      auto _parameters = dynamic_cast<cprg_parameter_pack*>(this->parameters);
      assert(_hyperparameters != nullptr);
      // still want to check whether or not to delete the existing stuff?

      std::vector<int> projection_set_size_threshold_ = {};
      if (projection_set_size_threshold_.size() < this->order){
        projection_set_size_threshold_.resize(this->order,0);// very small estimate
      }
      std::vector<std::vector<double>> valid_tensor_cell_points(this->order);
      int num_numerical_fm_rows=0;
      for (int i=0; i<this->numerical_modes.size(); i++){
        num_numerical_fm_rows += _parameters->num_partitions_per_dimension[this->numerical_modes[i]];
      }
      _parameters->spline_degree = _hyperparameters->max_spline_degree;
      double* temporary_extrap_models = new double[(1+_hyperparameters->max_spline_degree+1+_parameters->cp_rank)*num_numerical_fm_rows];
      num_numerical_fm_rows=0;//reset as index into temporary_extrap_models
      int fme_offset = 0;
      for (int i=0; i<this->order; i++){
        if (this->param_types[i] == parameter_type::CATEGORICAL) continue;
        // Inspect the FM to identify the projected set size
        int local_projected_set_size = 0;
        int max_elem = *std::max_element(this->Projected_Omegas[i].begin(),this->Projected_Omegas[i].end());
        for (int j=0; j<_parameters->num_partitions_per_dimension[i]; j++){
          assert(j<this->Projected_Omegas[i].size());
          if (this->Projected_Omegas[i][j] >= min(projection_set_size_threshold_[i],max_elem)){
            local_projected_set_size++;
            if (_hyperparameters->partition_spacing[i]==parameter_range_partition::SINGLE){
              valid_tensor_cell_points[i].push_back(_parameters->knot_positions[j+_parameters->knot_index_offsets[i]]);
            } else{
              assert(!(j+1 == _parameters->num_partitions_per_dimension[i]+1));
              valid_tensor_cell_points[i].push_back(get_midpoint_of_two_nodes(j,_parameters->num_partitions_per_dimension[i]+1,&_parameters->knot_positions[_parameters->knot_index_offsets[i]],_hyperparameters->partition_spacing[i]));
            }
          }
        }

        assert(local_projected_set_size>0);
        if (_parameters->cp_rank > local_projected_set_size ||
            (1+_parameters->spline_degree > local_projected_set_size)){
          //NOTE: If the assert is hit, then extrapolation likely won't be accurate, and you either need more samples or a smaller tensor and or smaller CP rank.
          int jump = 1+_parameters->spline_degree+1+_parameters->cp_rank;
          for (int k=0; k<jump; k++){
            temporary_extrap_models[num_numerical_fm_rows*jump+k] = 1.;
          }
          num_numerical_fm_rows++;
          fme_offset += _parameters->cp_rank*_parameters->num_partitions_per_dimension[i];
          continue;
        }

        assert(local_projected_set_size == _parameters->num_partitions_per_dimension[i]);
        local_projected_set_size = std::min(local_projected_set_size,_hyperparameters->max_training_set_size);
        double* reduced_matrix = new double[local_projected_set_size*_parameters->cp_rank];
        int column_count = 0;
        // reduced_matrix is column-major, as needed for LAPACK
        for (int j=_parameters->num_partitions_per_dimension[i]-local_projected_set_size; j<_parameters->num_partitions_per_dimension[i]; j++){
          assert(j<this->Projected_Omegas[i].size());
          if (this->Projected_Omegas[i][j] >= min(projection_set_size_threshold_[i],max_elem)){
            //std::cout << get_midpoint_of_two_nodes(j,_parameters->num_partitions_per_dimension[i]+(_hyperparameters->partition_spacing[i]==parameter_range_partition::SINGLE ? 0 : 1),&_parameters->knot_positions[_parameters->knot_index_offsets[i]],_hyperparameters->partition_spacing[i]) << ",";
            for (int k=0; k<_parameters->cp_rank; k++){
              reduced_matrix[column_count+local_projected_set_size*k] = _parameters->factor_matrix_elements[fme_offset+j*_parameters->cp_rank + k];
              //std::cout << reduced_matrix[column_count+local_projected_set_size*k] << ",";
            }
            //std::cout << "\n";
            column_count++;
          }
        }

        // Prep for a rank-1 SVD -- NOTE: We store all relevant columns/rows, not just low-rank factorization
        double* left_singular_matrix = new double[local_projected_set_size*_parameters->cp_rank];
        double* singular_value = new double[_parameters->cp_rank];
        double* right_singular_matrix = new double[_parameters->cp_rank*_parameters->cp_rank];

        double* superb_buffer = new double[5*local_projected_set_size];
        int svd_info;
        CTF_LAPACK::cdgesvd('S','S',local_projected_set_size,_parameters->cp_rank,reduced_matrix,local_projected_set_size,singular_value,left_singular_matrix,local_projected_set_size,right_singular_matrix,_parameters->cp_rank,superb_buffer,5*local_projected_set_size,&svd_info);
        assert(svd_info == 0);
        delete[] superb_buffer;

      // Curate the Perron vector: make it positive (which is always possible, see Thereom), but also restrict to constructing model solely out of strictly-increasing elements
      int num_elements_to_keep = local_projected_set_size;// should be <= local_projected_set_size
      //std::cout << "num_elements_to_keep - " << num_elements_to_keep << std::endl;
      for (int j=0; j<local_projected_set_size; j++){
         if (left_singular_matrix[j]<0) left_singular_matrix[j] *= (-1);
       }
       for (int j=0; j<_parameters->cp_rank; j++){
         if (right_singular_matrix[j*_parameters->cp_rank]<0) right_singular_matrix[j] = (-1)*right_singular_matrix[j*_parameters->cp_rank];
       }
       assert(singular_value[0]>0);

// Uncomment to get print out of Left Singular Vector of FM
/*
      std::cout << "Left SV\n";
      int kk=0;
      for (int j=_parameters->num_partitions_per_dimension[i]-local_projected_set_size; j<_parameters->num_partitions_per_dimension[i]; j++){
        std::cout << get_midpoint_of_two_nodes(j,_parameters->num_partitions_per_dimension[i]+(_hyperparameters->partition_spacing[i]==parameter_range_partition::SINGLE ? 0 : 1),&_parameters->knot_positions[_parameters->knot_index_offsets[i]],_hyperparameters->partition_spacing[i]) << "," << left_singular_matrix[kk++] << "\n";
      }
      std::cout << "End of Left SV\n";
*/
/*
      // Identify Perron vector: simply flip signs if necessary (of the first column only!)
      int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
      if (world_rank==0){
        std::cout << "Extrapolation with factor matrix " << i << ", which has size " << _parameters->num_partitions_per_dimension[i] << " " << _parameters->cp_rank << std::endl;
        std::cout << _parameters->num_partitions_per_dimension[0] << " " << _parameters->num_partitions_per_dimension[1] << " " << _parameters->num_partitions_per_dimension[2] << std::endl;
        for (int j=0; j<_parameters->num_partitions_per_dimension[0]; j++) std::cout << this->Projected_Omegas[0][j] << " ";
        std::cout << "\n";
        for (int j=0; j<_parameters->num_partitions_per_dimension[1]; j++) std::cout << this->Projected_Omegas[1][j] << " ";
        std::cout << "\n";
        for (int j=0; j<local_projected_set_size*_parameters->cp_rank; j++) std::cout << left_singular_matrix[j] << " ";
        std::cout << "\n";
        for (int j=0; j<_parameters->cp_rank; j++) std::cout << singular_value[j] << " ";
        std::cout << "\n";
        for (int j=0; j<_parameters->cp_rank*_parameters->cp_rank; j++) std::cout << right_singular_matrix[j] << " ";
        std::cout << "\n";
      }
*/

       assert(num_elements_to_keep <= local_projected_set_size);
       // Create simple linear model from ptr_to_data
       std::vector<double> feature_matrix(num_elements_to_keep*(1+_parameters->spline_degree),1.);
       for (int j=1; j<=_parameters->spline_degree; j++){
         for (int k=_parameters->num_partitions_per_dimension[i]-num_elements_to_keep; k<_parameters->num_partitions_per_dimension[i]; k++){
           //double midpoint_val = get_midpoint_of_two_nodes(k,local_projected_set_size+1,&_parameters->knot_positions[_parameters->knot_index_offsets[i]],this->interval_spacing[i]);
           assert(k<valid_tensor_cell_points[i].size());
           double point_val = valid_tensor_cell_points[i][k];
           int feature_matrix_idx = k-(_parameters->num_partitions_per_dimension[i]-num_elements_to_keep);
           feature_matrix[num_elements_to_keep*j+feature_matrix_idx] = feature_matrix[num_elements_to_keep*(j-1)+feature_matrix_idx] * (_hyperparameters->factor_matrix_underlying_position_transformation == parameter_transformation::LOG ? log(point_val) : point_val);
         }
       }
       for (int k=0; k<num_elements_to_keep; k++){
         left_singular_matrix[k] = _hyperparameters->factor_matrix_element_transformation==runtime_transformation::LOG ? log(left_singular_matrix[k]) : left_singular_matrix[k];
       }
/*
       if (world_rank == 0){
         std::cout << "Below is left singular matrix\n";
         for (int k=0; k<num_elements_to_keep; k++) std::cout << left_singular_matrix[k] << " ";
         std::cout << "\n";
         std::cout << "Below is feature matrix\n";
         for (int k=0; k<feature_matrix.size(); k++){
           std::cout << feature_matrix[k] << " ";
         }
         std::cout << "\n";
       }
*/
       //For simplicity sake, just solve for model coefficients via SVD
       int info;
       std::vector<double> work_buffer(1,-1);
       CTF_LAPACK::cdgels('N',num_elements_to_keep,1+_parameters->spline_degree,1,&feature_matrix[0],num_elements_to_keep,&left_singular_matrix[0],num_elements_to_keep,&work_buffer[0],-1,&info);
       assert(work_buffer[0]>0);
       int lwork = work_buffer[0];
       work_buffer.resize(lwork);
       CTF_LAPACK::cdgels('N',num_elements_to_keep,1+_parameters->spline_degree,1,&feature_matrix[0],num_elements_to_keep,&left_singular_matrix[0],num_elements_to_keep,&work_buffer[0],lwork,&info);
       assert(info==0);
       int jump = 1+_parameters->spline_degree+1+_parameters->cp_rank;
/*
       // Write model coefficients that fit the single left-singular vector
       for (int jjjj=0; jjjj<1+_parameters->spline_degree; jjjj++) std::cout << left_singular_matrix[jjjj] << " ";
       std::cout << singular_value[0] << " ";
       for (int jjjj=0; jjjj<_parameters->cp_rank; jjjj++) std::cout << right_singular_matrix[jjjj] << " ";
       std::cout << "\n";
*/
       std::memcpy(&temporary_extrap_models[num_numerical_fm_rows*jump],&left_singular_matrix[0],sizeof(double)*(1+_parameters->spline_degree));
       // Write single singular value
       std::memcpy(&temporary_extrap_models[num_numerical_fm_rows*jump+(1+_parameters->spline_degree)],&singular_value[0],sizeof(double));
       // Write single row of VT
       std::memcpy(&temporary_extrap_models[num_numerical_fm_rows*jump+(1+_parameters->spline_degree+1)],&right_singular_matrix[0],_parameters->cp_rank*sizeof(double));
       num_numerical_fm_rows++;
       fme_offset += _parameters->cp_rank*_parameters->num_partitions_per_dimension[i];
       delete[] reduced_matrix;
       delete[] left_singular_matrix;
       delete[] singular_value;
       delete[] right_singular_matrix;
   }
   //NOTE: Factor matrices are written as: tensor mode grows slowest, row of FM grows fastest

   _parameters->global_models = temporary_extrap_models;
   _parameters->num_models = num_numerical_fm_rows;
   this->m_is_valid=true;
   return true;
}

void cprg_model::write_to_file(const char* file_path) const{
  std::ofstream model_file_ptr;
  // Will overwrite anything in existing file
  model_file_ptr.open(file_path,std::ios_base::out);
  if(model_file_ptr.fail()) return;
  this->write_to_file(model_file_ptr);
  model_file_ptr.close();
}
void cprg_model::read_from_file(const char* file_path){
  std::ifstream model_file_ptr;
  // Will overwrite anything in existing file
  model_file_ptr.open(file_path,std::ios_base::in);
  if(model_file_ptr.fail()) return;
  this->read_from_file(model_file_ptr);
  model_file_ptr.close();
}
void cprg_model::write_to_file(std::ofstream& file) const{
  this->cpr_model::write_to_file(file);
}

void cprg_model::read_from_file(std::ifstream& file){
  // Nothing local to cpr_model to read
  this->cpr_model::read_from_file(file);
}
void cprg_model::get_hyperparameters(hyperparameter_pack& pack) const{
  this->hyperparameters->get(dynamic_cast<cprg_hyperparameter_pack&>(pack));
}
void cprg_model::set_hyperparameters(const hyperparameter_pack& pack){
  this->hyperparameters->set(dynamic_cast<const cpr_hyperparameter_pack&>(pack));
}
void cprg_model::get_parameters(parameter_pack& pack) const{
  this->parameters->get(dynamic_cast<cprg_parameter_pack&>(pack));
}

void cprg_model::set_parameters(const parameter_pack& pack){
  this->parameters->set(dynamic_cast<const cprg_parameter_pack&>(pack));
}

};
