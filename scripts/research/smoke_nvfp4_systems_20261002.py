"""Small SM120 native NVFP4 availability/correctness smoke; no performance claim.
Run with CUDA_VISIBLE_DEVICES=5 and the existing svdquant-ptq Python.
Packed codes are generated directly so this tests GEMM and SF layout separately
from an activation quantizer. No package installs or model loads are needed.
"""
import argparse
import json
from pathlib import Path
import torch


def swizzle(s):
    r, c = s.shape
    rp, cp = (r + 127) // 128 * 128, (c + 3) // 4 * 4
    p = torch.zeros((rp, cp), dtype=s.dtype, device=s.device)
    p[:r, :c] = s
    return p.view(rp // 128, 128, cp // 4, 4).permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


def operand(r, k):
    codes = torch.randint(0, 16, (r, k), dtype=torch.uint8, device='cuda')
    data = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.float4_e2m1fn_x2)
    # Nonuniform exactly representable scales exercise the hardware SF layout.
    scales = (torch.rand((r, k // 16), device='cuda') * 1.5 + 0.125).to(torch.float8_e4m3fn)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],device='cuda')
    ref = lut[codes.long()] * scales.float().repeat_interleave(16, dim=1)
    return data, swizzle(scales), ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {'torch':torch.__version__,'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'tests':[]}
    for m,n,k in [(128,128,256),(257,256,512)]:
        a, sa, ar = operand(m,k)
        b, sb, br = operand(n,k)
        rec = {'shape_mnk':[m,n,k]}
        try:
            y = torch._scaled_mm(a,b.t(),sa,sb,out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            exact = ar @ br.t()
            bf16_ref = ar.to(torch.bfloat16) @ br.to(torch.bfloat16).t()
            rec.update(status='ok',finite=bool(y.isfinite().all()),native_vs_exact_nmse=float((y.float()-exact).square().sum()/exact.square().sum()),native_vs_bf16_qdq_nmse=float((y.float()-bf16_ref.float()).square().sum()/bf16_ref.float().square().sum()),native_vs_bf16_qdq_maxabs=float((y.float()-bf16_ref.float()).abs().max()))
            if m == 128:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    z = torch._scaled_mm(a,b.t(),sa,sb,out_dtype=torch.bfloat16)
                    torch.cuda.synchronize()
                rec['cuda_kernel_names'] = sorted(set(e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA))
        except Exception as e:
            rec.update(status='error',error=repr(e))
        report['tests'].append(rec)
    # Exercise the real two-level recipe separately: global scales are not
    # rounded into each BF16 input before GEMM, unlike a BF16 QDQ surrogate.
    try:
        from torch.nn.functional import scaled_mm, ScalingType, SwizzleType
        a, sa, ar = operand(128,256)
        b, sb, br = operand(128,256)
        ga = torch.tensor([0.0137],device='cuda',dtype=torch.float32)
        gb = torch.tensor([0.00231],device='cuda',dtype=torch.float32)
        recipe=[ScalingType.BlockWise1x16,ScalingType.TensorWise]
        y = scaled_mm(a,b.t(),[sa,ga],recipe,[sb,gb],recipe,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=SwizzleType.SWIZZLE_32_4_4,output_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        exact=(ar @ br.t()) * ga * gb
        qdq=(ar * ga).to(torch.bfloat16) @ (br * gb).to(torch.bfloat16).t()
        report['two_level']={'status':'ok','global_scales':[float(ga),float(gb)],
            'native_vs_exact_nmse':float((y.float()-exact).square().sum()/exact.square().sum()),
            'native_vs_bf16_qdq_nmse':float((y.float()-qdq.float()).square().sum()/qdq.float().square().sum()),
            'fraction_different':float((y!=qdq).float().mean())}
    except Exception as e:
        report['two_level']={'status':'error','error':repr(e)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
