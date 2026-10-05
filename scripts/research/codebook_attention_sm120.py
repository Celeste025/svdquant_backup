#!/usr/bin/env python3
"""E033 private SM120 shared-correction-row consumer, D128/BF16/noncausal only.

Import and prepare_private_source() never query CUDA or compile. Only
get_codebook_module()/codebook_fwd() build the distinct private JIT module.
The caller must establish 0 <= centers_id < table.shape[2] outside timing;
validate_center_ids() is provided for preflight, not hidden in each forward.
"""
import argparse
import difflib
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
PRIVATE=Path('/data1/models/svdquant-wjq/research/private_jit/e033')
HEADER_PREFIX='flashinfer/attention/sm120/nvfp4_attention_sm120'


def file_record(path):
    path=Path(path).absolute();h=hashlib.sha256()
    with path.open('rb') as stream:
        for value in iter(lambda:stream.read(8*1024**2),b''):h.update(value)
    return dict(file=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())


def replace_once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)


def semantic_patches(original):
    """Four semantic files; namespace isolation is applied separately afterward."""
    result=dict(original)
    key='common/params.h';text=result[key]
    text=replace_once(text,'  void* __restrict__ delta_s_ptr;',
        '  void* __restrict__ delta_s_ptr;\n  int32_t const* __restrict__ centers_id_ptr;')
    result[key]=text
    key='api/launcher.h';text=result[key]
    text=replace_once(text,'       cutlass::FastDivmod(params.h_h_k_ratio),',
        '       params.centers_id_ptr,\n       cutlass::FastDivmod(params.h_h_k_ratio),')
    result[key]=text
    key='compute/mainloop.cuh';text=result[key]
    text=replace_once(text,'    StrideQKV const stride_ds;\n    cutlass::FastDivmod const group_size_fastdiv;',
        '    StrideQKV const stride_ds;\n    int32_t const* ptr_centers_id;\n    cutlass::FastDivmod const group_size_fastdiv;')
    text=replace_once(text,'    TMA_DS tma_load_DS;\n    cutlass::FastDivmod const group_size_fastdiv;',
        '    TMA_DS tma_load_DS;\n    int32_t const* ptr_centers_id;\n    cutlass::FastDivmod const group_size_fastdiv;')
    text=replace_once(text,'            tma_load_ds,\n            args.group_size_fastdiv,',
        '            tma_load_ds,\n            args.ptr_centers_id,\n            args.group_size_fastdiv,')
    text=replace_once(text,'                          make_coord(m_block, _));',
        '                          make_coord(mainloop_params.ptr_centers_id[\n'
        '                              (int64_t(bidb) * get<2>(mainloop_params.shape_Q) + bidh) *\n'
        '                                  (get<0>(mainloop_params.shape_Q) / 128) + m_block], _));')
    result[key]=text
    key='binding.cu';text=result[key]
    text=replace_once(text,'#include <flashinfer/attention/sm120/nvfp4_attention_sm120/api/launcher.h>',
        '#include "include/api/launcher.h"')
    text=replace_once(text,'constexpr DLDataType dl_uint8 = DLDataType{kDLUInt, 8, 1};',
        'constexpr DLDataType dl_uint8 = DLDataType{kDLUInt, 8, 1};\n'
        'constexpr DLDataType dl_int32 = DLDataType{kDLInt, 32, 1};')
    text=replace_once(text,'                      TensorView qk_correction, TensorView out, ffi::Optional<TensorView> maybe_lse,',
        '                      TensorView qk_correction, TensorView centers_id, TensorView out, ffi::Optional<TensorView> maybe_lse,')
    text=replace_once(text,'  params.delta_s_ptr = qk_correction.data_ptr();',
        '  params.delta_s_ptr = qk_correction.data_ptr();\n'
        '  params.centers_id_ptr = static_cast<int32_t const*>(centers_id.data_ptr());')
    text=replace_once(text,'  params.seqlen_s = per_block_mean ? seq_len_q : 128;',
        '  // DS logical M is independent of true Q M; zero-stride rows use K physical rows.\n'
        '  params.seqlen_s = static_cast<int>(qk_correction.size(2)) * 128;')
    start=text.index('template <bool ReturnLSE, bool IsBF16>')
    end=text.index('\n}  // namespace\n',start)
    # Instantiate only the already-existing D128/BF16/noncausal/BlockMean/noLSE branch.
    text=text[:start]+'''void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  using Traits = ::nvfp4_attention::Flash_fwd_kernel_traits<
      128, 128, 128, 3, 1, true,
      cutlass::nv_float4_t<cutlass::float_e2m1_t>, cutlass::bfloat16_t, float>;
  ::nvfp4_attention::run_flash_fwd<Traits, false, false>(params, stream);
}
'''+text[end:]
    text=replace_once(text,'         TensorView k_scale, TensorView v_scale_t, TensorView qk_correction, TensorView out,',
        '         TensorView k_scale, TensorView v_scale_t, TensorView qk_correction, TensorView centers_id, TensorView out,')
    text=replace_once(text,'  CHECK_INPUT(qk_correction);','  CHECK_INPUT(qk_correction);\n  CHECK_INPUT(centers_id);')
    text=replace_once(text,'  CHECK_DIM(4, qk_correction);','  CHECK_DIM(4, qk_correction);\n  CHECK_DIM(3, centers_id);')
    text=replace_once(text,'  TVM_FFI_ICHECK_EQ(qk_correction.dtype(), dl_float32) << "qk_correction must be float32";',
        '  TVM_FFI_ICHECK_EQ(qk_correction.dtype(), dl_float32) << "qk_correction must be float32";\n'
        '  TVM_FFI_ICHECK_EQ(centers_id.dtype(), dl_int32) << "centers_id must be int32";\n'
        '  TVM_FFI_ICHECK(!causal && per_block_mean && !maybe_lse.has_value());')
    text=replace_once(text,'  TVM_FFI_ICHECK(out.dtype() == dl_bfloat16 || out.dtype() == dl_float16)\n      << "out must be bfloat16 or float16";',
        '  TVM_FFI_ICHECK_EQ(out.dtype(), dl_bfloat16) << "E033 output must be bfloat16";')
    text=replace_once(text,'  TVM_FFI_ICHECK(head_dim == 64 || head_dim == 128) << "head_dim must be 64 or 128";',
        '  TVM_FFI_ICHECK_EQ(head_dim, 128);\n  TVM_FFI_ICHECK_EQ(batch, 1);\n'
        '  TVM_FFI_ICHECK_EQ(num_qo_heads, num_kv_heads);\n  TVM_FFI_ICHECK_GT(seq_len_q, 0);')
    text=replace_once(text,'  TVM_FFI_ICHECK_EQ(qk_correction.size(2), per_block_mean ? seq_len_q / 128 : 1);',
        '  TVM_FFI_ICHECK_GT(qk_correction.size(2), 0);\n'
        '  TVM_FFI_ICHECK_LE(qk_correction.size(2), INT32_MAX / 128);\n'
        '  TVM_FFI_ICHECK_EQ(centers_id.size(0), batch);\n'
        '  TVM_FFI_ICHECK_EQ(centers_id.size(1), num_qo_heads);\n'
        '  TVM_FFI_ICHECK_EQ(centers_id.size(2), seq_len_q / 128);')
    text=replace_once(text,'  check_same_device(q_fp4, qk_correction, "qk_correction");',
        '  check_same_device(q_fp4, qk_correction, "qk_correction");\n'
        '  check_same_device(q_fp4, centers_id, "centers_id");')
    text=replace_once(text,'  set_params_fprop(params, q_fp4, k_fp4, v_fp4_t, q_scale, k_scale, v_scale_t, qk_correction, out,',
        '  set_params_fprop(params, q_fp4, k_fp4, v_fp4_t, q_scale, k_scale, v_scale_t, qk_correction, centers_id, out,')
    text=replace_once(text,'TVM_FFI_DLL_EXPORT_TYPED_FUNC(fwd, flashinfer::nvfp4_attention_sm120::fwd);',
        'TVM_FFI_DLL_EXPORT_TYPED_FUNC(codebook_fwd, flashinfer::nvfp4_attention_sm120::fwd);')
    result[key]=text
    assert {key for key in original if original[key]!=result[key]}=={
        'common/params.h','api/launcher.h','compute/mainloop.cuh','binding.cu'}
    return result


def prepare_private_source():
    """Create/hash the small private source snapshot only; no CUDA query or JIT."""
    spec=importlib.util.find_spec('flashinfer');assert spec and spec.origin
    package=Path(spec.origin).parent
    headers=package/'data/include'/HEADER_PREFIX
    binding=package/'data/csrc/nvfp4_attention_sm120/nvfp4_attention_sm120_binding.cu'
    paths={str(path.relative_to(headers)):path for path in sorted(headers.rglob('*')) if path.is_file()}
    assert len(paths)==29,len(paths)
    paths['binding.cu']=binding
    original={key:path.read_text() for key,path in paths.items()}
    changed=semantic_patches(original)
    generator=file_record(__file__)
    identity=hashlib.sha256(json.dumps(dict(generator=generator['sha256'],sources=changed),sort_keys=True).encode()).hexdigest()
    namespace='e033_nvfp4_attention_'+identity[:16]
    private_dir=PRIVATE/identity
    generated={}
    for key,text in changed.items():
        # Pure symbol isolation, not algorithmic edits. Relative includes stay private.
        text=re.sub(r'\bnvfp4_attention\b',namespace,text)
        text=re.sub(r'\bnvfp4_attention_sm120\b',namespace+'_binding',text)
        text=re.sub(r'\bFlash_fwd_params\b',namespace+'_Flash_fwd_params',text)
        text=re.sub(r'\bQkv_params\b',namespace+'_Qkv_params',text)
        generated[key]=text
    patch=''.join(''.join(difflib.unified_diff(original[key].splitlines(True),changed[key].splitlines(True),
        fromfile='official/'+key,tofile='private_semantic/'+key)) for key in sorted(original) if original[key]!=changed[key])
    files={**{private_dir/'upstream'/key:text for key,text in original.items()},
           **{private_dir/('binding.cu' if key=='binding.cu' else 'include/'+key):text for key,text in generated.items()},
           private_dir/'semantic.patch':patch}
    for path,text in files.items():
        if path.exists():assert path.read_text()==text,str(path)
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    manifest=dict(source_hash=identity,jit_name='e033_nvfp4_codebook_'+identity[:16],namespace=namespace,
        private_dir=str(private_dir),binding=str(private_dir/'binding.cu'),generator=generator,
        official_header_count=29,official_sources={key:file_record(path) for key,path in paths.items()},
        generated_sources={key:file_record(private_dir/('binding.cu' if key=='binding.cu' else 'include/'+key)) for key in generated},
        patch=file_record(private_dir/'semantic.patch'),semantic_files=[key for key in original if original[key]!=changed[key]],
        structural_invariants=dict(unchanged_header_semantics=26,unchanged_tma_shared_barriers=True,
            true_q_shape_preserved=True,ds_extent='128 * correction_table.shape[2]',
            producer_row='ids[(batch * true_q_heads + head) * (true_q_length / 128) + m_block]',
            specialization='D128, BF16 output, noncausal, BlockMean=true, ReturnLSE=false; same original traits branch'),
        compilation_performed=False,cuda_queried=False)
    path=private_dir/'source_manifest.json'
    encoded=json.dumps(manifest,indent=2,sort_keys=True)+'\n'
    if path.exists():assert path.read_text()==encoded
    else:path.write_text(encoded)
    return manifest


source_manifest=prepare_private_source


@functools.cache
def get_codebook_module():
    """GPU-stage-only build, using original official flags and a distinct source name."""
    from flashinfer.jit import nvfp4_attention_sm120 as jit
    from flashinfer.jit.core import gen_jit_spec,current_compilation_context,sm120a_nvcc_flags
    manifest=prepare_private_source()
    try:arch=current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    except RuntimeError:arch=sm120a_nvcc_flags
    spec=gen_jit_spec(manifest['jit_name'],[Path(manifest['binding'])],
        extra_cuda_cflags=arch+jit._NVFP4_ATTENTION_SM120_CUDA_FLAGS,
        extra_include_paths=jit._nvfp4_attention_sm120_include_paths())
    return spec.build_and_load()


def validate_center_ids(centers_id,table_rows):
    """Explicit preflight; CUDA tensors synchronize here, never implicitly in fwd."""
    import torch
    if centers_id.dtype!=torch.int32 or centers_id.ndim!=3 or not centers_id.is_contiguous():
        raise ValueError('centers_id must be contiguous int32 [1,H,G]')
    if table_rows<=0 or not bool(((centers_id>=0)&(centers_id<table_rows)).all()):
        raise ValueError('center id outside correction table')


def check_metadata(q,k,v,qs,ks,vs,table,ids,*,out=None,unpadded_k_len=None,require_cuda=True):
    """No tensor value reads, CUDA queries or compilation; also usable on CPU/meta."""
    import torch
    tensors=(q,k,v,qs,ks,vs,table,ids)
    if any(not t.is_contiguous() or t.device!=q.device for t in tensors):
        raise ValueError('All inputs must be contiguous on the same device')
    if require_cuda and not q.is_cuda:raise ValueError('CUDA inputs required')
    if any(t.ndim!=4 for t in tensors[:-1]) or ids.ndim!=3:raise ValueError('Invalid tensor rank')
    b,h,m,d=q.shape;bk,hk,n,dk=k.shape
    if b!=1 or bk!=b or h<=0 or hk!=h or d!=64 or dk!=d or m<=0 or n<=0 or m%128 or n%128:
        raise ValueError('Only B1, equal Q/KV heads, positive padded M/N and D128 supported')
    expected=((b,h,128,n//2),(b,h,m,8),(b,h,n,8),(b,h,128,n//16))
    if any(tuple(t.shape)!=shape for t,shape in zip((v,qs,ks,vs),expected)):raise ValueError('Packet shape mismatch')
    if any(t.dtype!=torch.uint8 for t in (q,k,v)) or any(t.dtype!=torch.float8_e4m3fn for t in (qs,ks,vs)):
        raise ValueError('Expected packed uint8 codes and E4M3 scales')
    if table.dtype!=torch.float32 or table.shape[:2]!=(b,h) or table.shape[-1]!=n or not 0<table.shape[2]<=2147483647//128:
        raise ValueError('Expected FP32 correction table [1,H,K,Npad] with independent K')
    if ids.dtype!=torch.int32 or tuple(ids.shape)!=(b,h,m//128):raise ValueError('Expected int32 ids [1,H,Mpad/128]')
    logical=n if unpadded_k_len is None else unpadded_k_len
    if isinstance(logical,bool) or not isinstance(logical,int) or not 0<logical<=n:raise ValueError('Invalid logical K length')
    if out is not None and (out.dtype!=torch.bfloat16 or tuple(out.shape)!=(b,h,m,128) or out.device!=q.device or not out.is_contiguous()):
        raise ValueError('Output must be contiguous BF16 [1,H,Mpad,128] on the input device')
    return b,h,m,n,logical


def codebook_fwd(q_fp4,k_fp4,v_fp4_t,q_scale,k_scale,v_scale_t,correction_table,centers_id,
                 *,sm_scale=None,unpadded_k_len=None,out=None):
    """Trusted-ID fast path. Validate IDs once outside timing or construct via argmin(K).

    Same packet ABI as official FlashInfer; actual correction values are selected
    in its private producer. Q packing remains in the original official module.
    """
    import math
    import torch
    b,h,m,_,logical=check_metadata(q_fp4,k_fp4,v_fp4_t,q_scale,k_scale,v_scale_t,
        correction_table,centers_id,out=out,unpadded_k_len=unpadded_k_len)
    scale=128**-.5 if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale):raise ValueError('Scale must be finite')
    if out is None:out=torch.empty((b,h,m,128),device=q_fp4.device,dtype=torch.bfloat16)
    get_codebook_module().codebook_fwd(q_fp4,k_fp4,v_fp4_t,q_scale,k_scale,v_scale_t,
        correction_table,centers_id,out,None,scale,False,True,logical)
    return out


def cpu_check():
    import torch
    assert not torch.cuda.is_initialized()
    manifest=prepare_private_source()
    report=dict(status='complete',source= file_record(__file__),private_source=manifest,
                scope='CPU source/interface/row-mapping structure only; no native arithmetic or compilation evidence.')
    def meta(shape,dtype):return torch.empty(shape,dtype=dtype,device='meta')
    packets=[meta(shape,dtype) for shape,dtype in (
        ((1,56,22656,64),torch.uint8),((1,56,22656,64),torch.uint8),((1,56,128,11328),torch.uint8),
        ((1,56,22656,8),torch.float8_e4m3fn),((1,56,22656,8),torch.float8_e4m3fn),((1,56,128,1416),torch.float8_e4m3fn))]
    ids=meta((1,56,177),torch.int32)
    report['independent_table_extents']=[]
    for rows in (1,16,177):
        shape=check_metadata(*packets,meta((1,56,rows,22656),torch.float32),ids,
            unpadded_k_len=22539,require_cuda=False)
        assert shape==(1,56,22656,22656,22539)
        report['independent_table_extents'].append(dict(table_rows=rows,ds_logical_m=rows*128,true_q_length=shape[2],true_q_groups=177))
    old=json.loads((ROOT/'results/research/E032/run.json').read_text());assert old['status']=='complete'
    report['e032']=file_record(ROOT/'results/research/E032/run.json');report['actual_ids']=[]
    for row in old['layers']:
        ref=row['centers'];assert file_record(ref['file'])==ref
        value=torch.load(ref['file'],map_location='cpu',weights_only=True)
        actual=value['ids'].unsqueeze(0);validate_center_ids(actual,16)
        assert actual.shape==(1,56,177) and value['centers'].shape==(56,16,128)
        # Scalar row-table oracle checks flattened producer offset without large K tensors.
        table=torch.arange(56*16,dtype=torch.int64).reshape(1,56,16)
        gathered=table.gather(2,actual.long())
        loop=torch.empty_like(gathered)
        for head in range(56):
            for group in range(177):
                index=int(actual.reshape(-1)[head*177+group])
                loop[0,head,group]=table[0,head,index]
        assert torch.equal(gathered,loop)
        report['actual_ids'].append(dict(block=row['block'],artifact=ref,groups=actual.numel(),min=int(actual.min()),max=int(actual.max())))
    for identity in (False,True):
        mapping=torch.arange(177,dtype=torch.int32) if identity else torch.zeros(177,dtype=torch.int32)
        mapping=mapping[None,None].expand(1,56,177).contiguous()
        validate_center_ids(mapping,177 if identity else 1)
    assert not torch.cuda.is_initialized()
    report.update(cuda_initialized=False,compilation_performed=False)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/research/E033/consumer_cpu_check.json')
    args=parser.parse_args();assert not args.output.exists()
    import os
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    value=cpu_check();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],output=str(args.output),jit_name=value['private_source']['jit_name'])))


if __name__=='__main__':main()
