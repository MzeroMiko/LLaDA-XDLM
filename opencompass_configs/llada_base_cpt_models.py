from mmengine.config import read_base
import copy
import os

with read_base():
    from ..opencompass.opencompass.configs.models.dllm.llada_base_8b import \
        models as llada_base_8b_models


eval_cfg_llada = [
    {
        'gen_blocksize': 256, 
        'gen_length': 256, 
        'gen_steps': s,
        'abbr': f'llada-8b-base-l256-b256-s{s}',
    }
    for s in [32, 64, 128, 256]
]

eval_cfg_llada_k01 = [
    {
        'gen_blocksize': 256, 
        'gen_length': 256, 
        'gen_steps': s,
        'generation_kwargs': {'k1': 0.1},
        'abbr': f'llada-8b-base-l256-b256-s{s}-k01',
    }
    for s in [32, 64, 128, 256]
]

eval_cfg_schedhf_const_2em5_xdm_t600 = [
    {
        'gen_blocksize': 256, 
        'gen_length': 256, 
        'gen_steps': s,
        'generation_kwargs': {'k1': 0.1},
        'path': os.environ["ckpt_xdm"],
        'abbr': f'llada-8b-base-l256-b256-s{s}-k01-xdm-schedhf-const-2em5-t{ckpt}',
    }
    for ckpt in [600]
    for s in [32, 64, 128, 256]
]

eval_cfg_schedhf_const_2em5_mdm_t600 = [
    {
        'gen_blocksize': 256, 
        'gen_length': 256, 
        'gen_steps': s,
        'generation_kwargs': {'k1': 0.0},
        'path': os.environ["ckpt_mdm"],
        'abbr': f'llada-8b-base-l256-b256-s{s}-k00-mdm-schedhf-const-2em5-t{ckpt}',
    }
    for ckpt in [600]
    for s in [32, 64, 128, 256]
]

eval_cfgs = (
    eval_cfg_llada 
    + eval_cfg_llada_k01 
    + eval_cfg_schedhf_const_2em5_mdm_t600 
    + eval_cfg_schedhf_const_2em5_xdm_t600
)


models = [(copy.deepcopy(llada_base_8b_model) | cfg) for cfg in eval_cfgs for llada_base_8b_model in llada_base_8b_models]

