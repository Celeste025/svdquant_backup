#!/usr/bin/env python3
"""DeepCompressor PTQ entrypoint that swaps rCM's converted DiT into Wan."""
from __future__ import annotations
import gc, os, sys
from pathlib import Path
import torch

def main():
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
    from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
    from deepcompressor.app.diffusion.ptq import ptq
    from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
    from deepcompressor.app.diffusion.nn.patch import shift_input_activations
    from deepcompressor.app.diffusion.nn.struct import DiffusionAttentionStruct
    parser=DiffusionPtqRunConfig.get_parser(); config,*_=parser.parse_known_args()
    rcm=Path(os.environ['RCM_TRANSFORMER_PATH'])
    config.output.lock(); config.dump(path=config.output.get_running_job_path('config.yaml'))
    # Do not call config.pipeline.build(): its generic Wan build constructs the
    # base DiT before rCM replacement, which is incompatible with the local
    # Diffusers attention registration.  Construct the complete base pipeline,
    # install rCM first, then let DeepCompressor inspect the correct DiT.
    vae=AutoencoderKLWan.from_pretrained(config.pipeline.path,subfolder='vae',torch_dtype=torch.float32)
    pipe=WanPipeline.from_pretrained(config.pipeline.path,vae=vae,torch_dtype=torch.bfloat16)
    pipe.transformer=WanTransformer3DModel.from_pretrained(rcm,torch_dtype=torch.bfloat16)
    pipe=pipe.to('cuda')
    # Converted rCM checkpoints retain a WanAttention subclass which is not
    # present in the pinned Diffusers import used by DeepCompressor.  It has
    # the same Wan attention interface; register its concrete runtime class.
    rcm_attn_type=type(pipe.transformer.blocks[0].attn1)
    if rcm_attn_type not in DiffusionAttentionStruct._factories:
        DiffusionAttentionStruct.register_factory(rcm_attn_type, DiffusionAttentionStruct._default_construct)
    # Cache collection has the same exact-type whitelist for attention I/O.
    import deepcompressor.app.diffusion.dataset.calib as calib_module
    if rcm_attn_type not in calib_module._ATTN_TYPES:
        calib_module._ATTN_TYPES=(*calib_module._ATTN_TYPES, rcm_attn_type)
    if config.pipeline.shift_activations: shift_input_activations(pipe.transformer)
    for name in ('text_encoder','vae'):
        module=getattr(pipe,name,None)
        if isinstance(module,torch.nn.Module): module.to('cpu')
    gc.collect(); torch.cuda.empty_cache()
    model=DiffusionModelStruct.construct(pipe)
    save=str(config.save_model) if config.save_model and str(config.save_model).lower() not in ('false','none','null','nil') else ''
    ptq(model,config.quant,cache=config.cache,load_dirpath=config.load_from,save_dirpath=save,copy_on_save=config.copy_on_save,save_model=bool(save))
if __name__=='__main__': main()
