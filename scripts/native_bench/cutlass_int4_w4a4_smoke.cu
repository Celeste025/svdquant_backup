#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

using A = cutlass::int4b_t; using B = cutlass::int4b_t; using C = int32_t;
using LA = cutlass::layout::RowMajor; using LB = cutlass::layout::ColumnMajor; using LC = cutlass::layout::RowMajor;
using Epi = cutlass::epilogue::thread::LinearCombination<C, 4, C, C>;
using Gemm = cutlass::gemm::device::Gemm<A,LA,B,LB,C,LC,C,cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm75,cutlass::gemm::GemmShape<128,128,128>,cutlass::gemm::GemmShape<64,64,128>,
  cutlass::gemm::GemmShape<8,8,32>,Epi,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,2>;
int main() {
  cutlass::gemm::GemmCoord p(128,128,256); cutlass::HostTensor<A,LA>a(p.mk()); cutlass::HostTensor<B,LB>b(p.kn());
  cutlass::HostTensor<C,LC>c(p.mn()),d(p.mn()),r(p.mn());
  cutlass::reference::host::TensorFillRandomUniform(a.host_view(),1,A(4),A(-4),0);
  cutlass::reference::host::TensorFillRandomUniform(b.host_view(),2,B(4),B(-4),0);
  cutlass::reference::host::TensorFill(c.host_view()); a.sync_device();b.sync_device();c.sync_device();d.sync_device();r.sync_device();
  typename Gemm::Arguments args{p,a.device_ref(),b.device_ref(),c.device_ref(),d.device_ref(),{1,0},1};
  Gemm op; auto st=op.can_implement(args); if(st!=cutlass::Status::kSuccess){std::cerr<<"can_implement "<<int(st)<<"\n";return 2;}
  cutlass::device_memory::allocation<uint8_t> ws(Gemm::get_workspace_size(args));
  if(op.initialize(args,ws.get())!=cutlass::Status::kSuccess || op()!=cutlass::Status::kSuccess){std::cerr<<"launch failure\n";return 3;}
  cutlass::reference::device::Gemm<A,LA,B,LB,C,LC,C,C> ref; ref(p,1,a.device_ref(),b.device_ref(),0,c.device_ref(),r.device_ref());
  cudaError_t e=cudaDeviceSynchronize(); if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<"\n";return 4;} d.sync_host();r.sync_host();
  bool ok=cutlass::reference::host::TensorEquals(d.host_view(),r.host_view()); std::cout<<(ok?"PASS":"FAIL")<<" legacy_sm75_int4xint4\n";return ok?0:5;
}
