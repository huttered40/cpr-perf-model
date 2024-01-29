/*
	Adapted from: https://github.com/solomonik/APSP/blob/master/fmm/dgemm_esolomon.cxx
*/

#include "string.h"
#include "stdio.h"
#include <immintrin.h>
//#include <xmmintrin.h>
//#include <x86intrin.>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#define VECTOR_WIDTH 8	// AVX-512

#ifndef RBN
#define RBN 1
#endif

#ifndef RBK
#define RBK 2
#endif

#ifndef RBM
#define RBM 2
#endif

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

/*
#if (RBK==2)
  #define INIT_ROW_A(ib) \
    __m128d A_##ib##_0; 
  #define INIT_ROW_B(jb) \
    __m128d B_##jb##_0; 
  #define LOAD_ROW_A(ib) \
    A_##ib##_0 = _mm_load_pd(A+k+0+(i+(ib))*ldk);
  #define LOAD_ROW_B(jb) \
    B_##jb##_0 = _mm_load_pd(B+k+0+(j+(jb))*ldk);
  #define MUL_ROW_A_B(ib, jb) \
    C_##jb##_##ib = _mm_add_pd(C_##jb##_##ib,_mm_mul_pd(A_##ib##_0,B_##jb##_0)); 
#endif

#if (RBK==4)
  #define INIT_ROW_A(ib) \
    __m128d A_##ib##_0; \
    __m128d A_##ib##_1; 
  #define INIT_ROW_B(jb) \
    __m128d B_##jb##_0; \
    __m128d B_##jb##_1; 
  #define LOAD_ROW_A(ib) \
    A_##ib##_0 = ((__m128d *)(A+k+0+(i+(ib))*ldk))[0]; \
    A_##ib##_1 = ((__m128d *)(A+k+2+(i+(ib))*ldk))[0];
  #define LOAD_ROW_B(jb) \
    B_##jb##_0 = ((__m128d *)(B+k+0+(j+(jb))*ldk))[0]; \
    B_##jb##_1 = ((__m128d *)(B+k+2+(j+(jb))*ldk))[0];
  #define MUL_ROW_A_B(ib, jb) \
    C_##jb##_##ib = _mm_add_pd(C_##jb##_##ib, _mm_mul_pd(A_##ib##_0,B_##jb##_0)); \
    C_##jb##_##ib = _mm_add_pd(C_##jb##_##ib, _mm_mul_pd(A_##ib##_1,B_##jb##_1));
#endif    

#if (RBK==8)
  #define INIT_ROW_A(ib) \
    __m128d A_##ib##_0; __m128d A_##ib##_1; \
    __m128d A_##ib##_2; __m128d A_##ib##_3; 
  #define INIT_ROW_B(jb) \
    __m128d B_##jb##_0; __m128d B_##jb##_1; \
    __m128d B_##jb##_2; __m128d B_##jb##_3; 
  #define LOAD_ROW_A(ib) \
    A_##ib##_0 = ((__m128d *)(A+k+0+(i+(ib))*ldk))[0]; \
    A_##ib##_1 = ((__m128d *)(A+k+2+(i+(ib))*ldk))[0]; \
    A_##ib##_2 = ((__m128d *)(A+k+4+(i+(ib))*ldk))[0]; \
    A_##ib##_3 = ((__m128d *)(A+k+6+(i+(ib))*ldk))[0];
  #define LOAD_ROW_B(jb) \
    B_##jb##_0 = ((__m128d *)(B+k+0+(j+(jb))*ldk))[0]; \
    B_##jb##_1 = ((__m128d *)(B+k+2+(j+(jb))*ldk))[0]; \
    B_##jb##_2 = ((__m128d *)(B+k+4+(j+(jb))*ldk))[0]; \
    B_##jb##_3 = ((__m128d *)(B+k+6+(j+(jb))*ldk))[0];

  #define MUL_ROW_A_B(ib, jb) \
    C_##jb##_##ib = _mm_add_pd(C_##jb##_##ib, _mm_mul_pd(A_##ib##_0,B_##jb##_0)); \
    C_##jb##_##ib = _mm_add_pd(C_##jb##_##ib, _mm_mul_pd(A_##ib##_1,B_##jb##_1)); \
    C_##jb##_##ib = _mm_add_pd(C_##jb##_##ib, _mm_mul_pd(A_##ib##_2,B_##jb##_2)); \
    C_##jb##_##ib = _mm_add_pd(C_##jb##_##ib, _mm_mul_pd(A_##ib##_3,B_##jb##_3));
#endif    


#if (RBN==1)
  #define INIT_B \
    INIT_ROW_B(0)
  #define LOAD_B \
    LOAD_ROW_B(0)
  
  #define INIT_ROW_C(ib) \
    __m128d C_0_##ib; \
    double C1_0_##ib; \
    double C2_0_##ib; 
  #define LOAD_ROW_C(ib) \
    C_0_##ib = _mm_setzero_pd();
       
  #define STORE_ROW_C(ib) \
    _mm_storeh_pd(&C1_0_##ib,C_0_##ib);\
    _mm_storel_pd(&C2_0_##ib,C_0_##ib);\
    C[j+0+(i+(ib))*ldn] += C1_0_##ib + C2_0_##ib;

  #define MUL_SQUARE_A_B(ib) \
    MUL_ROW_A_B(ib, 0)
#endif

#if (RBN==2)
  #define INIT_B \
    INIT_ROW_B(0) INIT_ROW_B(1)
  #define LOAD_B \
    LOAD_ROW_B(0) LOAD_ROW_B(1) 
  
  #define INIT_ROW_C(ib) \
    __m128d C_0_##ib; __m128d C_1_##ib; \
    __m128d C_swap_##ib;

  #define LOAD_ROW_C(ib) \
    C_0_##ib = _mm_setzero_pd(); \
    C_1_##ib = _mm_setzero_pd();

  #define STORE_ROW_C(ib) \
    C_swap_##ib = _mm_unpacklo_pd(C_0_##ib, C_1_##ib); \
    C_0_##ib = _mm_unpackhi_pd(C_0_##ib, C_1_##ib); \
    C_1_##ib = _mm_load_pd(C+j+0+(i+(ib))*ldn); \
    C_1_##ib = _mm_add_pd(C_1_##ib,C_0_##ib); \
    C_1_##ib = _mm_add_pd(C_1_##ib,C_swap_##ib); \
    _mm_store_pd(C+j+0+(i+(ib))*ldn, C_1_##ib); 

  #define MUL_SQUARE_A_B(ib) \
    MUL_ROW_A_B(ib, 0) MUL_ROW_A_B(ib, 1)
#endif

#if (RBN==4)
  #define INIT_B \
    INIT_ROW_B(0) INIT_ROW_B(1) \
    INIT_ROW_B(2) INIT_ROW_B(3)
  #define LOAD_B \
    LOAD_ROW_B(0) LOAD_ROW_B(1) \
    LOAD_ROW_B(2) LOAD_ROW_B(3) 
  
  #define INIT_ROW_C(ib) \
    __m128d C_0_##ib; __m128d C_1_##ib; \
    __m128d C_swap_0_##ib; \
    __m128d C_2_##ib; __m128d C_3_##ib; \
    __m128d C_swap_1_##ib; 
  #define LOAD_ROW_C(ib) \
    C_0_##ib = _mm_setzero_pd(); \
    C_1_##ib = _mm_setzero_pd(); \
    C_2_##ib = _mm_setzero_pd(); \
    C_3_##ib = _mm_setzero_pd();

  #define STORE_ROW_C(ib) \
    C_swap_0_##ib = _mm_unpacklo_pd(C_0_##ib, C_1_##ib); \
    C_0_##ib = _mm_unpackhi_pd(C_0_##ib, C_1_##ib); \
    C_1_##ib = _mm_load_pd(C+j+0+(i+(ib))*ldn); \
    C_1_##ib = _mm_add_pd(C_1_##ib,C_0_##ib); \
    C_1_##ib = _mm_add_pd(C_1_##ib,C_swap_0_##ib); \
    _mm_store_pd(C+j+0+(i+(ib))*ldn, C_1_##ib); \
    C_swap_1_##ib = _mm_unpacklo_pd(C_2_##ib, C_3_##ib); \
    C_2_##ib = _mm_unpackhi_pd(C_2_##ib, C_3_##ib); \
    C_3_##ib = _mm_load_pd(C+j+2+(i+(ib))*ldn); \
    C_3_##ib = _mm_add_pd(C_3_##ib,C_2_##ib); \
    C_3_##ib = _mm_add_pd(C_3_##ib,C_swap_1_##ib); \
    _mm_store_pd(C+j+2+(i+(ib))*ldn, C_3_##ib); 


  #define MUL_SQUARE_A_B(ib) \
    MUL_ROW_A_B(ib, 0) MUL_ROW_A_B(ib, 1) \
    MUL_ROW_A_B(ib, 2) MUL_ROW_A_B(ib, 3)
#endif

#if (RBN==8)
  #define INIT_B \
    INIT_ROW_B(0) INIT_ROW_B(1) \
    INIT_ROW_B(2) INIT_ROW_B(3) \
    INIT_ROW_B(4) INIT_ROW_B(5) \
    INIT_ROW_B(6) INIT_ROW_B(7)
  #define LOAD_B \
    LOAD_ROW_B(0) LOAD_ROW_B(1) \
    LOAD_ROW_B(2) LOAD_ROW_B(3) \
    LOAD_ROW_B(4) LOAD_ROW_B(5) \
    LOAD_ROW_B(6) LOAD_ROW_B(7) 
  
  #define INIT_ROW_C(ib) \
    __m128d C_0_##ib; __m128d C_1_##ib; \
    __m128d C_swap_0_##ib; \
    __m128d C_2_##ib; __m128d C_3_##ib; \
    __m128d C_swap_1_##ib; \
    __m128d C_4_##ib; __m128d C_5_##ib; \
    __m128d C_swap_2_##ib; \
    __m128d C_6_##ib; __m128d C_7_##ib; \
    __m128d C_swap_3_##ib; 
  #define LOAD_ROW_C(ib) \
    C_0_##ib = _mm_setzero_pd(); \
    C_1_##ib = _mm_setzero_pd(); \
    C_2_##ib = _mm_setzero_pd(); \
    C_3_##ib = _mm_setzero_pd(); \
    C_4_##ib = _mm_setzero_pd(); \
    C_5_##ib = _mm_setzero_pd(); \
    C_6_##ib = _mm_setzero_pd(); \
    C_7_##ib = _mm_setzero_pd();

  #define STORE_ROW_C(ib) \
    C_swap_0_##ib = _mm_unpacklo_pd(C_0_##ib, C_1_##ib); \
    C_0_##ib = _mm_unpackhi_pd(C_0_##ib, C_1_##ib); \
    C_1_##ib = _mm_load_pd(C+j+0+(i+(ib))*ldn); \
    C_1_##ib = _mm_add_pd(C_1_##ib,C_0_##ib); \
    C_1_##ib = _mm_add_pd(C_1_##ib,C_swap_0_##ib); \
    _mm_store_pd(C+j+0+(i+(ib))*ldn, C_1_##ib); \
    C_swap_1_##ib = _mm_unpacklo_pd(C_2_##ib, C_3_##ib); \
    C_2_##ib = _mm_unpackhi_pd(C_2_##ib, C_3_##ib); \
    C_3_##ib = _mm_load_pd(C+j+2+(i+(ib))*ldn); \
    C_3_##ib = _mm_add_pd(C_3_##ib,C_2_##ib); \
    C_3_##ib = _mm_add_pd(C_3_##ib,C_swap_1_##ib); \
    _mm_store_pd(C+j+2+(i+(ib))*ldn, C_3_##ib); \
    C_swap_2_##ib = _mm_unpacklo_pd(C_4_##ib, C_5_##ib); \
    C_4_##ib = _mm_unpackhi_pd(C_4_##ib, C_5_##ib); \
    C_5_##ib = _mm_load_pd(C+j+4+(i+(ib))*ldn); \
    C_5_##ib = _mm_add_pd(C_5_##ib,C_4_##ib); \
    C_5_##ib = _mm_add_pd(C_5_##ib,C_swap_2_##ib); \
    _mm_store_pd(C+j+4+(i+(ib))*ldn, C_5_##ib); \
    C_swap_3_##ib = _mm_unpacklo_pd(C_6_##ib, C_7_##ib); \
    C_6_##ib = _mm_unpackhi_pd(C_6_##ib, C_7_##ib); \
    C_7_##ib = _mm_load_pd(C+j+6+(i+(ib))*ldn); \
    C_7_##ib = _mm_add_pd(C_7_##ib,C_6_##ib); \
    C_7_##ib = _mm_add_pd(C_7_##ib,C_swap_3_##ib); \
    _mm_store_pd(C+j+6+(i+(ib))*ldn, C_7_##ib); 


  #define MUL_SQUARE_A_B(ib) \
    MUL_ROW_A_B(ib, 0) MUL_ROW_A_B(ib, 1) \
    MUL_ROW_A_B(ib, 2) MUL_ROW_A_B(ib, 3) \
    MUL_ROW_A_B(ib, 4) MUL_ROW_A_B(ib, 5) \
    MUL_ROW_A_B(ib, 6) MUL_ROW_A_B(ib, 7)
#endif


#if (RBM==1)
  #define INIT_A \
    INIT_ROW_A(0) 
  #define LOAD_A \
    LOAD_ROW_A(0)
  #define INIT_C \
    INIT_ROW_C(0) 
  #define LOAD_C \
    LOAD_ROW_C(0)
  #define STORE_C \
    STORE_ROW_C(0)
  #define MUL_A_B \
    MUL_SQUARE_A_B(0) 
#endif

#if (RBM==2)
  #define INIT_A \
    INIT_ROW_A(0) INIT_ROW_A(1) 
  #define LOAD_A \
    LOAD_ROW_A(0) LOAD_ROW_A(1)
  #define INIT_C \
    INIT_ROW_C(0) INIT_ROW_C(1) 
  #define LOAD_C \
    LOAD_ROW_C(0) LOAD_ROW_C(1)
  #define STORE_C \
    STORE_ROW_C(0) STORE_ROW_C(1)
  #define MUL_A_B \
    MUL_SQUARE_A_B(0) MUL_SQUARE_A_B(1) 
#endif

#if (RBM==4)
  #define INIT_A \
    INIT_ROW_A(0) INIT_ROW_A(1) \
    INIT_ROW_A(2) INIT_ROW_A(3) 
  #define LOAD_A \
    LOAD_ROW_A(0) LOAD_ROW_A(1) \
    LOAD_ROW_A(2) LOAD_ROW_A(3)
  #define INIT_C \
    INIT_ROW_C(0) INIT_ROW_C(1) \
    INIT_ROW_C(2) INIT_ROW_C(3) 
  #define LOAD_C \
    LOAD_ROW_C(0) LOAD_ROW_C(1) \
    LOAD_ROW_C(2) LOAD_ROW_C(3)
  #define STORE_C \
    STORE_ROW_C(0) STORE_ROW_C(1) \
    STORE_ROW_C(2) STORE_ROW_C(3)
  #define MUL_A_B \
    MUL_SQUARE_A_B(0) MUL_SQUARE_A_B(1) \
    MUL_SQUARE_A_B(2) MUL_SQUARE_A_B(3) 
#endif

#if (RBM==8)
  #define INIT_A \
    INIT_ROW_A(0) INIT_ROW_A(1) \
    INIT_ROW_A(2) INIT_ROW_A(3) \
    INIT_ROW_A(4) INIT_ROW_A(5) \
    INIT_ROW_A(6) INIT_ROW_A(7) 
  #define LOAD_A \
    LOAD_ROW_A(0) LOAD_ROW_A(1) \
    LOAD_ROW_A(2) LOAD_ROW_A(3) \
    LOAD_ROW_A(4) LOAD_ROW_A(5) \
    LOAD_ROW_A(6) LOAD_ROW_A(7)
  #define INIT_C \
    INIT_ROW_C(0) INIT_ROW_C(1) \
    INIT_ROW_C(2) INIT_ROW_C(3) \
    INIT_ROW_C(4) INIT_ROW_C(5) \
    INIT_ROW_C(6) INIT_ROW_C(7) 
  #define LOAD_C \
    LOAD_ROW_C(0) LOAD_ROW_C(1) \
    LOAD_ROW_C(2) LOAD_ROW_C(3) \
    LOAD_ROW_C(4) LOAD_ROW_C(5) \
    LOAD_ROW_C(6) LOAD_ROW_C(7)
  #define STORE_C \
    STORE_ROW_C(0) STORE_ROW_C(1) \
    STORE_ROW_C(2) STORE_ROW_C(3) \
    STORE_ROW_C(4) STORE_ROW_C(5) \
    STORE_ROW_C(6) STORE_ROW_C(7)
  #define MUL_A_B \
    MUL_SQUARE_A_B(0) MUL_SQUARE_A_B(1) \
    MUL_SQUARE_A_B(2) MUL_SQUARE_A_B(3) \
    MUL_SQUARE_A_B(4) MUL_SQUARE_A_B(5) \
    MUL_SQUARE_A_B(6) MUL_SQUARE_A_B(7) 
#endif
*/
#if (RBK==8)
  #define INIT_ROW_A(ib) \
    __m512d A_##ib##_0; 
  #define INIT_ROW_B(jb) \
    __m512d B_##jb##_0; 
  #define LOAD_ROW_A(ib) \
    A_##ib##_0 = _mm512_load_pd(A+k+0+(i+(ib))*ldk);
  #define LOAD_ROW_B(jb) \
    B_##jb##_0 = _mm512_load_pd(B+k+0+(j+(jb))*ldk);
  #define MUL_ROW_A_B(ib, jb) \
    C_##jb##_##ib = _mm512_add_pd(C_##jb##_##ib,_mm512_mul_pd(A_##ib##_0,B_##jb##_0)); 
#endif

#if (RBK==16)
  #define INIT_ROW_A(ib) \
    __m512d A_##ib##_0; __m512d A_##ib##_1; 
  #define INIT_ROW_B(jb) \
    __m512d B_##jb##_0; __m512d B_##jb##_1; 
  #define LOAD_ROW_A(ib) \
    A_##ib##_0 = ((__m512d *)(A+k+0+(i+(ib))*ldk))[0]; \
    A_##ib##_1 = ((__m512d *)(A+k+8+(i+(ib))*ldk))[0];
  #define LOAD_ROW_B(jb) \
    B_##jb##_0 = ((__m512d *)(B+k+0+(j+(jb))*ldk))[0]; \
    B_##jb##_1 = ((__m512d *)(B+k+8+(j+(jb))*ldk))[0];
  #define MUL_ROW_A_B(ib, jb) \
    C_##jb##_##ib = _mm512_add_pd(C_##jb##_##ib, _mm512_mul_pd(A_##ib##_0,B_##jb##_0)); \
    C_##jb##_##ib = _mm512_add_pd(C_##jb##_##ib, _mm512_mul_pd(A_##ib##_1,B_##jb##_1));
#endif    

#if (RBK==24)
  #define INIT_ROW_A(ib) \
    __m512d A_##ib##_0; __m512d A_##ib##_1; __m512d A_##ib##_2; 
  #define INIT_ROW_B(jb) \
    __m512d B_##jb##_0; __m512d B_##jb##_1; __m512d B_##jb##_2;
  #define LOAD_ROW_A(ib) \
    A_##ib##_0 = ((__m512d *)(A+k+0+(i+(ib))*ldk))[0]; \
    A_##ib##_1 = ((__m512d *)(A+k+8+(i+(ib))*ldk))[0]; \
    A_##ib##_2 = ((__m512d *)(A+k+16+(i+(ib))*ldk))[0]; \
  #define LOAD_ROW_B(jb) \
    B_##jb##_0 = ((__m512d *)(B+k+0+(j+(jb))*ldk))[0]; \
    B_##jb##_1 = ((__m512d *)(B+k+8+(j+(jb))*ldk))[0]; \
    B_##jb##_2 = ((__m512d *)(B+k+16+(j+(jb))*ldk))[0]; \

  #define MUL_ROW_A_B(ib, jb) \
    C_##jb##_##ib = _mm512_add_pd(C_##jb##_##ib, _mm512_mul_pd(A_##ib##_0,B_##jb##_0)); \
    C_##jb##_##ib = _mm512_add_pd(C_##jb##_##ib, _mm512_mul_pd(A_##ib##_1,B_##jb##_1)); \
    C_##jb##_##ib = _mm512_add_pd(C_##jb##_##ib, _mm512_mul_pd(A_##ib##_2,B_##jb##_2));
#endif    

#if (RBK==32)
  #define INIT_ROW_A(ib) \
    __m512d A_##ib##_0; __m512d A_##ib##_1; __m512d A_##ib##_2; __m512d A_##ib##_3; 
  #define INIT_ROW_B(jb) \
    __m512d B_##jb##_0; __m512d B_##jb##_1; __m512d B_##jb##_2; __m512d B_##jb##_3; 
  #define LOAD_ROW_A(ib) \
    A_##ib##_0 = ((__m512d *)(A+k+0+(i+(ib))*ldk))[0]; \
    A_##ib##_1 = ((__m512d *)(A+k+8+(i+(ib))*ldk))[0]; \
    A_##ib##_2 = ((__m512d *)(A+k+16+(i+(ib))*ldk))[0]; \
    A_##ib##_3 = ((__m512d *)(A+k+24+(i+(ib))*ldk))[0];
  #define LOAD_ROW_B(jb) \
    B_##jb##_0 = ((__m512d *)(B+k+0+(j+(jb))*ldk))[0]; \
    B_##jb##_1 = ((__m512d *)(B+k+8+(j+(jb))*ldk))[0]; \
    B_##jb##_2 = ((__m512d *)(B+k+16+(j+(jb))*ldk))[0]; \
    B_##jb##_3 = ((__m512d *)(B+k+24+(j+(jb))*ldk))[0];

  #define MUL_ROW_A_B(ib, jb) \
    C_##jb##_##ib = _mm512_add_pd(C_##jb##_##ib, _mm512_mul_pd(A_##ib##_0,B_##jb##_0)); \
    C_##jb##_##ib = _mm512_add_pd(C_##jb##_##ib, _mm512_mul_pd(A_##ib##_1,B_##jb##_1)); \
    C_##jb##_##ib = _mm512_add_pd(C_##jb##_##ib, _mm512_mul_pd(A_##ib##_2,B_##jb##_2)); \
    C_##jb##_##ib = _mm512_add_pd(C_##jb##_##ib, _mm512_mul_pd(A_##ib##_3,B_##jb##_3));
#endif    


#if (RBN==1)
  #define INIT_B \
    INIT_ROW_B(0)
  #define LOAD_B \
    LOAD_ROW_B(0)
  
  #define INIT_ROW_C(ib) \
    __m512d C_0_##ib;
  #define LOAD_ROW_C(ib) \
    C_0_##ib = _mm512_setzero_pd();
       
  #define STORE_ROW_C(ib) \
    C[j+0+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_0_##ib);

  #define MUL_SQUARE_A_B(ib) \
    MUL_ROW_A_B(ib, 0)
#endif

#if (RBN==2)
  #define INIT_B \
    INIT_ROW_B(0) INIT_ROW_B(1)
  #define LOAD_B \
    LOAD_ROW_B(0) LOAD_ROW_B(1) 
  
  #define INIT_ROW_C(ib) \
    __m512d C_0_##ib; __m512d C_1_##ib; \

  #define LOAD_ROW_C(ib) \
    C_0_##ib = _mm512_setzero_pd(); \
    C_1_##ib = _mm512_setzero_pd();

  #define STORE_ROW_C(ib) \
    C[j+0+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_0_##ib); \
    C[j+1+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_1_##ib);

  #define MUL_SQUARE_A_B(ib) \
    MUL_ROW_A_B(ib, 0) MUL_ROW_A_B(ib, 1)
#endif

#if (RBN==4)
  #define INIT_B \
    INIT_ROW_B(0) INIT_ROW_B(1) INIT_ROW_B(2) INIT_ROW_B(3)
  #define LOAD_B \
    LOAD_ROW_B(0) LOAD_ROW_B(1) LOAD_ROW_B(2) LOAD_ROW_B(3) 
  
  #define INIT_ROW_C(ib) \
    __m512d C_0_##ib; __m512d C_1_##ib; \
    __m512d C_2_##ib; __m512d C_3_##ib;
  #define LOAD_ROW_C(ib) \
    C_0_##ib = _mm512_setzero_pd(); \
    C_1_##ib = _mm512_setzero_pd(); \
    C_2_##ib = _mm512_setzero_pd(); \
    C_3_##ib = _mm512_setzero_pd();

  #define STORE_ROW_C(ib) \
    C[j+0+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_0_##ib); \
    C[j+1+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_1_##ib); \
    C[j+2+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_2_##ib); \
    C[j+3+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_3_##ib);

  #define MUL_SQUARE_A_B(ib) \
    MUL_ROW_A_B(ib, 0) MUL_ROW_A_B(ib, 1) MUL_ROW_A_B(ib, 2) MUL_ROW_A_B(ib, 3)
#endif

#if (RBN==8)
  #define INIT_B \
    INIT_ROW_B(0) INIT_ROW_B(1) INIT_ROW_B(2) INIT_ROW_B(3) \
    INIT_ROW_B(4) INIT_ROW_B(5) INIT_ROW_B(6) INIT_ROW_B(7)
  #define LOAD_B \
    LOAD_ROW_B(0) LOAD_ROW_B(1) LOAD_ROW_B(2) LOAD_ROW_B(3) \
    LOAD_ROW_B(4) LOAD_ROW_B(5) LOAD_ROW_B(6) LOAD_ROW_B(7) 
  
  #define INIT_ROW_C(ib) \
    __m512d C_0_##ib; __m512d C_1_##ib; \
    __m512d C_swap_0_##ib; \
    __m512d C_2_##ib; __m512d C_3_##ib; \
    __m512d C_swap_1_##ib; \
    __m512d C_4_##ib; __m512d C_5_##ib; \
    __m512d C_swap_2_##ib; \
    __m512d C_6_##ib; __m512d C_7_##ib; \
    __m512d C_swap_3_##ib; 
  #define LOAD_ROW_C(ib) \
    C_0_##ib = _mm512_setzero_pd(); \
    C_1_##ib = _mm512_setzero_pd(); \
    C_2_##ib = _mm512_setzero_pd(); \
    C_3_##ib = _mm512_setzero_pd(); \
    C_4_##ib = _mm512_setzero_pd(); \
    C_5_##ib = _mm512_setzero_pd(); \
    C_6_##ib = _mm512_setzero_pd(); \
    C_7_##ib = _mm512_setzero_pd();

  #define STORE_ROW_C(ib) \
    C[j+0+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_0_##ib); \
    C[j+1+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_1_##ib); \
    C[j+2+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_2_##ib); \
    C[j+3+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_3_##ib); \
    C[j+4+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_4_##ib); \
    C[j+5+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_5_##ib); \
    C[j+6+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_6_##ib); \
    C[j+7+(i+(ib))*ldn] += _mm512_reduce_add_pd(C_7_##ib);

  #define MUL_SQUARE_A_B(ib) \
    MUL_ROW_A_B(ib, 0) MUL_ROW_A_B(ib, 1) MUL_ROW_A_B(ib, 2) MUL_ROW_A_B(ib, 3) \
    MUL_ROW_A_B(ib, 4) MUL_ROW_A_B(ib, 5) MUL_ROW_A_B(ib, 6) MUL_ROW_A_B(ib, 7)
#endif


#if (RBM==1)
  #define INIT_A \
    INIT_ROW_A(0) 
  #define LOAD_A \
    LOAD_ROW_A(0)
  #define INIT_C \
    INIT_ROW_C(0) 
  #define LOAD_C \
    LOAD_ROW_C(0)
  #define STORE_C \
    STORE_ROW_C(0)
  #define MUL_A_B \
    MUL_SQUARE_A_B(0) 
#endif

#if (RBM==2)
  #define INIT_A \
    INIT_ROW_A(0) INIT_ROW_A(1) 
  #define LOAD_A \
    LOAD_ROW_A(0) LOAD_ROW_A(1)
  #define INIT_C \
    INIT_ROW_C(0) INIT_ROW_C(1) 
  #define LOAD_C \
    LOAD_ROW_C(0) LOAD_ROW_C(1)
  #define STORE_C \
    STORE_ROW_C(0) STORE_ROW_C(1)
  #define MUL_A_B \
    MUL_SQUARE_A_B(0) MUL_SQUARE_A_B(1) 
#endif

#if (RBM==4)
  #define INIT_A \
    INIT_ROW_A(0) INIT_ROW_A(1) \
    INIT_ROW_A(2) INIT_ROW_A(3) 
  #define LOAD_A \
    LOAD_ROW_A(0) LOAD_ROW_A(1) \
    LOAD_ROW_A(2) LOAD_ROW_A(3)
  #define INIT_C \
    INIT_ROW_C(0) INIT_ROW_C(1) \
    INIT_ROW_C(2) INIT_ROW_C(3) 
  #define LOAD_C \
    LOAD_ROW_C(0) LOAD_ROW_C(1) \
    LOAD_ROW_C(2) LOAD_ROW_C(3)
  #define STORE_C \
    STORE_ROW_C(0) STORE_ROW_C(1) \
    STORE_ROW_C(2) STORE_ROW_C(3)
  #define MUL_A_B \
    MUL_SQUARE_A_B(0) MUL_SQUARE_A_B(1) \
    MUL_SQUARE_A_B(2) MUL_SQUARE_A_B(3) 
#endif

#if (RBM==8)
  #define INIT_A \
    INIT_ROW_A(0) INIT_ROW_A(1) \
    INIT_ROW_A(2) INIT_ROW_A(3) \
    INIT_ROW_A(4) INIT_ROW_A(5) \
    INIT_ROW_A(6) INIT_ROW_A(7) 
  #define LOAD_A \
    LOAD_ROW_A(0) LOAD_ROW_A(1) \
    LOAD_ROW_A(2) LOAD_ROW_A(3) \
    LOAD_ROW_A(4) LOAD_ROW_A(5) \
    LOAD_ROW_A(6) LOAD_ROW_A(7)
  #define INIT_C \
    INIT_ROW_C(0) INIT_ROW_C(1) \
    INIT_ROW_C(2) INIT_ROW_C(3) \
    INIT_ROW_C(4) INIT_ROW_C(5) \
    INIT_ROW_C(6) INIT_ROW_C(7) 
  #define LOAD_C \
    LOAD_ROW_C(0) LOAD_ROW_C(1) \
    LOAD_ROW_C(2) LOAD_ROW_C(3) \
    LOAD_ROW_C(4) LOAD_ROW_C(5) \
    LOAD_ROW_C(6) LOAD_ROW_C(7)
  #define STORE_C \
    STORE_ROW_C(0) STORE_ROW_C(1) \
    STORE_ROW_C(2) STORE_ROW_C(3) \
    STORE_ROW_C(4) STORE_ROW_C(5) \
    STORE_ROW_C(6) STORE_ROW_C(7)
  #define MUL_A_B \
    MUL_SQUARE_A_B(0) MUL_SQUARE_A_B(1) \
    MUL_SQUARE_A_B(2) MUL_SQUARE_A_B(3) \
    MUL_SQUARE_A_B(4) MUL_SQUARE_A_B(5) \
    MUL_SQUARE_A_B(6) MUL_SQUARE_A_B(7) 
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////


inline static void blk_transp(double const* block,
                              double* new_block,
                              int nrow,
                              int ncol){
  for (int i = 0; i < nrow; i++){
    for (int j = 0; j < ncol; j++){
      new_block[i*ncol+j]=block[j*nrow+i];
    }
  }
}

#pragma safeptr=all
inline static void do_block(int ldm,
                            int ldn,
                            int ldk,
                            int M, 
                            int N, 
                            int K, 
                            int BLOCK_SIZE_M, 
                            int BLOCK_SIZE_N, 
                            int BLOCK_SIZE_K, 
                            double const* A, 
                            double const* B,
                            double* C,
                            int const full_KN){
// To optimize this, think about loop unrolling and software
//     pipelining.  Hint:  For the majority of the matmuls, you
//         know exactly how many iterations there are (the block size)...

  int const mm = (M/RBM + (M%RBM > 0))*RBM;
  int const nn = (N/RBN + (N%RBN > 0))*RBN;
  int const kk = (K/RBK + (K%RBK > 0))*RBK;

  int i,j,k;

  INIT_A
  INIT_B
  INIT_C

  if (full_KN == 3){
    for (i = 0; i < mm; i+= RBM){
      #pragma unroll BLOCK_SIZE_N/RBN
      for (j = 0; j < nn; j+= RBN){
        LOAD_C;
        #pragma unroll BLOCK_SIZE_K/RBK
        for (k = 0; k < BLOCK_SIZE_K; k+= RBK){
          LOAD_A;
          LOAD_B;
          MUL_A_B;
        }
        STORE_C;
      }
    }
  } else if (full_KN == 2){
    for (i = 0; i < mm; i+= RBM){
      for (j = 0; j < nn; j+= RBN){
        LOAD_C;
        #pragma unroll BLOCK_SIZE_K/RBK
        for (k = 0; k < BLOCK_SIZE_K; k+= RBK){
          LOAD_A;
          LOAD_B;
          MUL_A_B;
        }
        STORE_C;
      }
    }
  } else{
    for (i = 0; i < mm; i+= RBM){
      for (j = 0; j < nn; j+= RBN){
        LOAD_C;
        #pragma unroll
        for (k = 0; k < kk; k+= RBK){
          LOAD_A;
          LOAD_B;
          MUL_A_B;
        }
        STORE_C;
      }
    }
  }
}

std::chrono::duration<double> rect_dgemm(int m, int n, int k, 
                                         int L2M, int L2N, int L2K, 
                                         int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, 
                                         double *A, double *B, double *C, bool verify=false)
{
  int ldm_pad = m;
  int ldn_pad = n;
  int ldk_pad = k;
  if (ldm_pad % RBM != 0) ldm_pad = (ldm_pad/RBM)*RBM + RBM;
  if (ldn_pad % RBN != 0) ldn_pad = (ldn_pad/RBN)*RBN + RBN;
  if (ldk_pad % RBK != 0) ldk_pad = (ldk_pad/RBK)*RBK + RBK;

  // ldapad may be slightly larger than lda, but can now handle k-way unrolling
  double pad_A[ldm_pad*ldk_pad] __attribute__ ((aligned(64)));
  double A_swap[ldm_pad*ldk_pad] __attribute__ ((aligned(64)));
  double pad_B[ldk_pad*ldn_pad] __attribute__ ((aligned(64)));
  double pad_C[ldm_pad*ldn_pad] __attribute__ ((aligned(64)));
  double C_swap[ldm_pad*ldn_pad] __attribute__ ((aligned(64)));
  for (int i = 0; i< k; i++){
    memcpy(pad_A+i*ldm_pad, A+i*m, m*sizeof(double));
    std::fill(pad_A+i*ldm_pad+m, pad_A+(i+1)*ldm_pad, 0);
  }
  for (int i = k; i< ldk_pad; i++){
    std::fill(pad_A+i*ldm_pad, pad_A+(i+1)*ldm_pad, 0);
  }
  for (int i = 0; i< n; i++){
    memcpy(pad_B+i*ldk_pad, B+i*k, k*sizeof(double));
    std::fill(pad_B+i*ldk_pad+k, pad_B+(i+1)*ldk_pad, 0);
  }
  for (int i = n; i< ldn_pad; i++){
    std::fill(pad_B+i*ldk_pad, pad_B+(i+1)*ldk_pad, 0);
  }
  std::fill(pad_C, pad_C+ldm_pad*ldn_pad, 0);

  auto start_time = std::chrono::high_resolution_clock::now();
/*
  //NOTE: We comment this out, but it is needed for correctness
  blk_transp(pad_A, A_swap, ldm_pad, ldk_pad);
*/

/*
  // Verify that blk_transp is correct
  if (verify){
    for (int i=0; i<m; i++){
       for (int j=0; j<k; j++){
         std::cout << pad_A[i+j*ldm_pad] << " ";// this is how pad_A is accessed  (strided accesss)
       }
       std::cout << "\n";
    }
    std::cout << "\n\n\n";
    for (int i=0; i<m; i++){
       for (int j=0; j<k; j++){
         std::cout << A_swap[i*ldk_pad+j] << " ";// the whole point is for the fastest dimension to resemble the slowest dimension in the other
       }
       std::cout << "\n";
    }
  }
*/

  /*For each block combination*/
  #pragma omp parallel for collapse(2)
  for( int i2 = 0; i2 < ldm_pad; i2 += L2M ) {
    for( int j2 = 0; j2 < ldn_pad; j2 += L2N ) {
      for( int k2 = 0; k2 < ldk_pad; k2 += L2K ) {
	for( int i = i2; i < min(ldm_pad,i2+L2M); i += BLOCK_SIZE_M ) {
	  for( int j = j2; j < min(ldn_pad,j2+L2N); j += BLOCK_SIZE_N ) {
	    for( int kk = k2; kk < min(ldk_pad,k2+L2K); kk += BLOCK_SIZE_K ) {
	      /*This gets the correct block size (for fringe blocks also)*/
	      int M = min( BLOCK_SIZE_M, ldm_pad-i );
	      int N = min( BLOCK_SIZE_N, ldn_pad-j );
	      int K = min( BLOCK_SIZE_K, ldk_pad-kk );
	      do_block(ldm_pad, ldn_pad, ldk_pad, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, 
		       A_swap+kk+i*ldk_pad, pad_B+kk+j*ldk_pad, pad_C+j+i*ldn_pad, 
		       2*(K==BLOCK_SIZE_K)+(N==BLOCK_SIZE_N));
/*
              // Verify that do_block is correct
              std::cout << "Do block " << i << " " << j << " " << kk << " " << ldm_pad << " " << ldn_pad << " " << ldk_pad << " " << L2M << " " << L2N << " " << L2K << "\n";
              if (verify){
		// Let's check accuracy to be sure
		for (int ii1=0; ii1<m; ii1++){
		  for (int ii2=0; ii2<n; ii2++){
		    double verify = 0.;
		    for (int ii3=0; ii3<k; ii3++){
                      //std::cout << "What are these - (" << ii1 << " " << ii2 << " " << ii3 << "): " << A_swap[ii3+ii1*k] << " " << pad_B[ii3+ii2*k] << std::endl;
		      verify += A_swap[ii3+ii1*k]*pad_B[ii3+ii2*k];
		    }
		    //std::cout << ii1 << "," << ii2 << ": " << verify << " " << pad_C[ii1*n+ii2] << " " << abs(verify - pad_C[ii1*n+ii2])/verify << " " << std::endl;
		    assert(abs(verify - pad_C[ii1*n+ii2])/verify < 1e-15);
		  }
		}
              }
*/
	    }
	  }
	}
      }
    }
  }
/*
  //NOTE: We comment this out, but it is needed for correctness
  blk_transp(pad_C, C_swap, ldn_pad, ldm_pad);
  for (int i = 0; i< n; i++){
    memcpy(C+i*m,C_swap+i*ldm_pad, m*sizeof(double));
  }
*/
  // array C stores the final output
  auto end = std::chrono::high_resolution_clock::now();

  if (verify){
    // Let's check accuracy to be sure
    for (int i=0; i<m; i++){
      for (int ii=0; ii<n; ii++){
        double verify = 0.;
        for (int iii=0; iii<k; iii++){
          verify += A[iii*m+i]*B[iii+ii*k];
        }
        //std::cout << i << "," << ii << ": " << verify << " " << C[ii*m+i] << " " << abs(verify - C[ii*m+i])/verify << " " << std::endl;
        assert(abs(verify - C[ii*m+i])/verify < 1e-15);
      }
    }
  }
  // C stored in row-major
  return end - start_time;
}

int main( int argc, char** argv )
{
    int min_iter=10;// This must be set by user to generate a distinct binary
    int max_iter=50;// This must be set by user to generate a distinct binary
    double rel_std_dev_tol=0.01;// This must be set by user to generate a distinct binary
    std::string file_location = "/work2/05608/tg849075/cpr-perf-model/datasets/generation_scripts/kernel_gemm_multiblocked/datafiles/";

    double delta,msq,mean,std_dev,rel_std_dev;
    int m = (argc>1 ? atoi(argv[1]) : 2048);
    int n = (argc>2 ? atoi(argv[2]) : 2048);
    int k = (argc>3 ? atoi(argv[3]) : 2048);
    int outer_cache_block_size_m = (argc>4 ? atoi(argv[4]) : min(256,m));
    int outer_cache_block_size_n = (argc>5 ? atoi(argv[5]) : min(256,n));
    int outer_cache_block_size_k = (argc>6 ? atoi(argv[6]) : min(256,k));
    int inner_cache_block_size_m = (argc>7 ? max(RBM,atoi(argv[7])) : min(32,outer_cache_block_size_m));
    int inner_cache_block_size_n = (argc>8 ? max(RBN,atoi(argv[8])) : min(32,outer_cache_block_size_n));
    int inner_cache_block_size_k = (argc>9 ? max(RBK,atoi(argv[9])) : min(32,outer_cache_block_size_k));
    int register_block_size_m = RBM;
    int register_block_size_n = RBN;
    int register_block_size_k = RBK;
    int nthreads = (argc>10 ? atoi(argv[10]) : 1);
    int node_id = (argc>11 ? atoi(argv[11]) : 0);
    bool verify = (argc>12 ? atoi(argv[12]) : false);
    std::string machine_name = std::getenv("MACHINE_NAME");
    std::ofstream write_file;
    std::string write_str = file_location + "multi-blocked-gemm-" + machine_name + "-" + std::to_string(nthreads) + "-" + std::to_string(node_id) + ".csv";
    write_file.open(write_str,std::fstream::app);//,std::ios_base::app);

    // Generate matrices
    srand48(100);
    std::vector<double> A(m*k,1.);
    std::vector<double> B(k*n,1.);
    std::vector<double> C(m*n,1.);
    for (int i=0; i<m*k; i++) A[i] = drand48();
    for (int i=0; i<k*n; i++) B[i] = drand48();
    // Warm-up with a cold-start execution
    auto forget_this = rect_dgemm(m,n,k,outer_cache_block_size_m,outer_cache_block_size_n,outer_cache_block_size_k,inner_cache_block_size_m,inner_cache_block_size_n,inner_cache_block_size_k,&A[0],&B[0],&C[0],verify);
    mean=0.; msq=0; delta=0;
    int j=0;
    while (1){
      auto diff = rect_dgemm(m,n,k,outer_cache_block_size_m,outer_cache_block_size_n,outer_cache_block_size_k,inner_cache_block_size_m,inner_cache_block_size_n,inner_cache_block_size_k,&A[0],&B[0],&C[0],false);
      delta = diff.count() - mean; mean += delta/(j+1); msq += delta*(diff.count()-mean);
      ++j;
      std_dev = j>1 ? sqrt(msq/(j-1)) : 100000.; rel_std_dev = std_dev/mean;
      if (j>= min_iter && rel_std_dev < rel_std_dev_tol) break; else if (j>=max_iter) break;
    }
    write_file << 0 << "," << m << "," << n << "," << k << ","
                           << outer_cache_block_size_m << "," << outer_cache_block_size_n << "," << outer_cache_block_size_k << ","
                           << inner_cache_block_size_m << "," << inner_cache_block_size_n << "," << inner_cache_block_size_k << ","
                           << register_block_size_m << "," << register_block_size_n << "," << register_block_size_k << ","
                           << nthreads << "," << mean << "," << rel_std_dev << "," << j << std::endl;
   return 0;
}
