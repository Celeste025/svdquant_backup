// Minimal cuBLASLt INT4xINT4 tensor-core smoke test.
// This intentionally uses packed CUDA_R_4I operands, not QDQ BF16 operands.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); return 2;} } while(0)
#define CHECK_LT(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){std::fprintf(stderr,"cuBLASLt status %d\n",int(s)); return 3;} } while(0)

int main(int argc, char** argv) {
  int m = argc > 1 ? std::atoi(argv[1]) : 128;
  int n = argc > 2 ? std::atoi(argv[2]) : 128;
  int k = argc > 3 ? std::atoi(argv[3]) : 256;
  if ((m*k) % 2 || (k*n) % 2) return 4;
  cublasLtHandle_t handle; CHECK_LT(cublasLtCreate(&handle));
  cublasLtMatmulDesc_t op;
  CHECK_LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  cublasLtMatrixLayout_t a,b,c,d;
  // Column-major A[M,K] and B[K,N]; each byte holds two signed INT4 values.
  CHECK_LT(cublasLtMatrixLayoutCreate(&a, CUDA_R_4I, m, k, m));
  CHECK_LT(cublasLtMatrixLayoutCreate(&b, CUDA_R_4I, k, n, k));
  CHECK_LT(cublasLtMatrixLayoutCreate(&c, CUDA_R_32I, m, n, m));
  CHECK_LT(cublasLtMatrixLayoutCreate(&d, CUDA_R_32I, m, n, m));
  void *A,*B,*C,*D,*ws; size_t ws_bytes=8<<20;
  CHECK_CUDA(cudaMalloc(&A, size_t(m)*k/2)); CHECK_CUDA(cudaMalloc(&B, size_t(k)*n/2));
  CHECK_CUDA(cudaMalloc(&C, size_t(m)*n*4)); CHECK_CUDA(cudaMalloc(&D, size_t(m)*n*4)); CHECK_CUDA(cudaMalloc(&ws,ws_bytes));
  CHECK_CUDA(cudaMemset(A, 0x11, size_t(m)*k/2)); CHECK_CUDA(cudaMemset(B, 0x11, size_t(k)*n/2)); CHECK_CUDA(cudaMemset(C, 0, size_t(m)*n*4));
  cublasLtMatmulPreference_t pref; CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_LT(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws_bytes,sizeof(ws_bytes)));
  cublasLtMatmulHeuristicResult_t heuristic{}; int returned=0;
  CHECK_LT(cublasLtMatmulAlgoGetHeuristic(handle,op,a,b,c,d,pref,1,&heuristic,&returned));
  if (!returned) { std::fprintf(stderr,"no INT4xINT4 algorithm\n"); return 5; }
  int alpha=1,beta=0;
  CHECK_LT(cublasLtMatmul(handle,op,&alpha,A,a,B,b,&beta,C,c,D,d,&heuristic.algo,ws,ws_bytes,0));
  CHECK_CUDA(cudaDeviceSynchronize());
  int first=0; CHECK_CUDA(cudaMemcpy(&first,D,sizeof(first),cudaMemcpyDeviceToHost));
  std::printf("PASS m=%d n=%d k=%d output0=%d expected=%d workspace=%zu\n",m,n,k,first,k,heuristic.workspaceSize);
  cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(D); cudaFree(ws);
  cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b); cublasLtMatrixLayoutDestroy(c); cublasLtMatrixLayoutDestroy(d); cublasLtMatmulDescDestroy(op); cublasLtDestroy(handle);
  return first == k ? 0 : 6;
}
