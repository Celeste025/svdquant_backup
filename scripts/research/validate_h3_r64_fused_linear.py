import json,sys,torch,torch.nn.functional as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/research'),'/data1/models/svdquant-wjq/research/deps/cutlass-dsl-4.6.2-cu13','/data1/models/svdquant-wjq/research/deps/cutlass-dsl-4.6.2-cu13/nvidia_cutlass_dsl/dsl_packages']
from h3_nvfp4_zero_sf_compat import pack_activation_fast
from flashinfer.gemm import mm_nvfp4_svdquant
p=torch.load('/data1/models/svdquant-wjq/research/20261102/h3_svdquant_rank64_native_export/fused_inputs/blocks.0.attn.qkv_proj.pt',map_location='cpu',weights_only=True,mmap=True)
x=p['x_s'][:256].cuda().contiguous();a=p['a'].cuda();b=p['b'].cuda();w=p['weight_packed'].cuda();ws=p['weight_scales_swizzled'].cuda();wg=p['weight_global'].cuda();packet=pack_activation_fast(x);down=F.linear(x,a);alpha=(packet.global_scale*wg).reshape(1);l=(b.float()/alpha).bfloat16().contiguous()
def fused():return mm_nvfp4_svdquant(packet.packed,w,packet.swizzled_scales.view(torch.uint8).reshape(-1),ws.view(torch.uint8).reshape(-1),alpha,down,l,backend='cute-dsl',enable_pdl=False)
def reference():
 r=[F.ScalingType.BlockWise1x16,F.ScalingType.TensorWise]
 y=F.scaled_mm(packet.packed.view(torch.float4_e2m1fn_x2),w.view(torch.float4_e2m1fn_x2).t(),[packet.swizzled_scales,packet.global_scale],r,[ws,wg],r,swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,output_dtype=torch.bfloat16)
 return y+F.linear(down,b)
y=fused();z=reference();torch.cuda.synchronize();d=y.float()-z.float();result={'rank':64,'layer':'blocks.0.attn.qkv_proj','tokens':256,'finite':bool(torch.isfinite(y).all()),'nmse_vs_native_unfused':float(d.square().sum()/z.float().square().sum()),'max_abs_error':float(d.abs().max()),'backend':'cute-dsl SM120 fused NVFP4 GEMM plus LoRA up','source':p['raw_input_sha256']};assert result['finite'] and result['nmse_vs_native_unfused']<0.001
out=ROOT/'results/research/h3_r64_fused/fused_real_input_validation.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
