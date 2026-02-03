<div align=center>
  
# [miXed Diffusion Language Modeling](https://arxiv.org/pdf/2602.01362)
</div>

This is the official implementaion of paper [***Balancing Understanding and Generation in Discrete Diffusion Models***](https://arxiv.org/pdf/2602.01362). This repository contains Pytorch training and evaluation code for ***continual pretraining LLaDA with XDLM***.

## LLaDA Continue Pretraining

***LLaDA-XDLM with sampling budget of 32.***
Evaluation of adapting LLaDA-8B to our XDLM formulation (LLaDA-XDLM): (a) LLaDA-XDLM consistently out-performs baselines across diverse benchmarks with 32 sampling steps; (b) Improvements are particularly pronounced in code generation (MBPP), where the
model substantially reduces generation failures.
<div align=center>
<img src="docs/xdlm_llada.png" width="80%">
</div>

## Preparations

### Prepare Environment for Training
```bash
POSTFIX="-i https://mirrors.ustc.edu.cn/pypi/simple"
NAME="xdlm"
eval "$(conda shell.bash hook)";
conda tos accept;
conda create -n $NAME python==3.12 -y && conda activate $NAME;
which conda; which python; which pip;
pip install --resume-retries 999 datasets==2.15.0 einops==0.7.0 fsspec git-lfs==1.6 h5py==3.10.0 hydra-core==1.3.2 ipdb==0.13.13 lightning==2.2.1 notebook==7.1.1 nvitop==1.3.2 omegaconf==2.3.0 packaging==23.2 pandas==2.2.1 rich==13.7.1 seaborn==0.13.2 scikit-learn==1.4.0 transformers==4.38.2 triton==2.2.0 torch==2.3.1 torchaudio==2.3.1 torchmetrics==1.6.1 torchvision==0.18.1 wandb timm ocifs hf_transfer huggingface-hub mauve-text==0.4.0 pytorch-image-generation-metrics==0.6.1 torch_fidelity==0.3.0 deepspeed==0.13.1 evaluate peft==0.10.0 accelerate==0.27.2 $POSTFIX;
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.6/flash_attn-2.7.4.post1+cu126torch2.3-cp312-cp312-linux_x86_64.whl;
```

### Prepare Environment for Evaluating
```bash
POSTFIX="-i https://mirrors.ustc.edu.cn/pypi/simple"
NAME="lladaop"
eval "$(conda shell.bash hook)";
conda tos accept;
conda create -n $NAME python==3.10 -y && conda activate $NAME;
which conda; which python; which pip;
pip install -r requirement.txt $POSTFIX;
cd opencompass && pip install -e . $POSTFIX && cd ..
git clone https://github.com/open-compass/human-eval.git
cd human-eval && pip install -e . $POSTFIX && cd ..
```

## Training
```bash
deepspeed llada_cpt.py \
    --deepspeed ds_zero2_bf16_config_simp.json \
    --config_name "GSAI-ML/LLaDA-8B-Base" \
    --tokenizer_name "GSAI-ML/LLaDA-8B-Base" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_name ${HF_DATASETS_CACHE}/HuggingFaceFW___fineweb-edu \
    --dataset_config_name sample-10BT \
    --streaming \
    --block_size 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --output_dir ./checkpoints \
    --overwrite_output_dir \
    --bf16 \
    --gradient_accumulation_steps 64 \
    --lr_scheduler_type "constant_with_warmup" \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    --logging_steps 2 \
    --save_steps 100 \
    --max_steps 1000 \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 16 \
    --report_to wandb \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --load_best_model_at_end False \
    --training_method "xdm" \
    --trust_remote_code True
```

## Evaluation
***difference between `opencompass` vs `LLaDA/opencompass`***: we add `**self.generation_kwargs` in opencompass/opencompass/models/dllm.py#LLaDA_generate to enable inputing custom generation kwargs.

```bash
python opencompass/run.py opencompass_configs/llada_base_gen_gsm8k_length256_block256.py -w outputs/llada_base_gsm8k_length256_block256;
python opencompass/run.py opencompass_configs/llada_base_gen_math_length256_block256.py -w outputs/llada_base_math_length256_block256;
python opencompass/run.py opencompass_configs/llada_base_gen_humaneval_length256_block256.py -w outputs/llada_base_humaneval_length256_block256;
python opencompass/run.py opencompass_configs/llada_base_gen_mbpp_length256_block256.py -w outputs/llada_base_mbpp_length256_block256;
python opencompass/run.py opencompass_configs/llada_base_gen_bbh_length256_block256.py -w outputs/llada_base_bbh_length256_block256;
```


## Contacts
If you have any question about our work or this repository, please don't hesitate to contact us by emails or open an issue under this project.
- [liuyue171@mails.ucas.ac.cn](liuyue171@mails.ucas.ac.cn)
- [zhaoyuzhong20@mails.ucas.ac.cn](zhaoyuzhong20@mails.ucas.ac.cn)
- [caoshaosheng@xiaohongshu.com](caoshaosheng@xiaohongshu.com)
- [liuyunfan@ucas.ac.cn](liuyunfan@ucas.ac.cn)

## Acknowledgment

- The main training code is build off of [transformers](https://github.com/huggingface/transformers), and the evaluation code is build off of [LLaDA](https://github.com/ML-GSAI/LLaDA).
we sincerely thank them for their contributions to the community.

## Citation

coming soon


