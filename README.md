# FFB: Fast and Furiour Bitsandbytes

FFB is a more effecient implmentation for the popular quantization method - Bitsandbits (LLM.int8()). It parallizes matrix multiplication and reduces GPU context management overhead. 

Original Github Repository:
- [Bitsandbytes Code](https://github.com/TimDettmers/bitsandbytes)

Resources:
- [8-bit Optimizer Paper](https://arxiv.org/abs/2110.02861) --  [Video](https://www.youtube.com/watch?v=IxrlHAJtqKE) -- [Docs](https://bitsandbytes.readthedocs.io/en/latest/)

- [LLM.int8() Paper](https://arxiv.org/abs/2208.07339) -- [LLM.int8() Software Blog Post](https://huggingface.co/blog/hf-bitsandbytes-integration) -- [LLM.int8() Emergent Features Blog Post](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)

## Requirements & Installation

**Recommended requirements:** 
- python==3.9
- pytorch==1.13.1
- torchvision==0.14.1
- torchaudio==0.13.1
- pytorch-cuda==11.6
- transformers==4.33.3
- tokenizers==0.13.3
- accelerate==0.20.3
- scipy==1.10.1

(This is the environment used for the final project evaluation. A different library version may lead to compatibility issue.)

The requirements can best be fulfilled by installing pytorch via anaconda. You can install PyTorch by following the ["Get Started"](https://pytorch.org/get-started/locally/) instructions on the official website. 

For a standard FFB installation via anaconda, follow the quickstart:

```bash
conda create --name FFB python=3.9

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda==11.6 -c pytorch -c nvidia

conda install -c huggingface transformers==4.33.3

conda install -c conda-forge tokenizers==0.13.3

conda install -c conda-forge accelerate==0.20.3

conda install scipy==1.10.1
```

**Hardware requirements:** NVIDIA Turing (RTX 20xx; T4) or Ampere GPU (RTX 30xx; A4-A100); (a GPU from 2018 or older).

(Make sure you make enough GPU memory!!!)

**Installation:** run the following commands to compile FFB from source: 

```bash
# CUDA_VERSIONS in {110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 120}
# make argument in {cuda110, cuda11x, cuda12x}
CUDA_VERSION=116 make cuda11x
python setup.py install
```

## Using FFB

FFB does not change the interface of Bitandbytes. It does not require additional modifications. A updated detailed example can be found in [examples/int8_inference_huggingface.py](examples/int8_inference_huggingface.py).

### Using 8-bit inference:
1. Comment out torch.nn.Linear: ``#linear = torch.nn.Linear(...)``
2. Add bnb 8-bit linear light module: ``linear = bnb.nn.Linear8bitLt(...)`` (base arguments stay the same)
3. There are two modes:
   - Mixed 8-bit training with 16-bit main weights. Pass the argument ``has_fp16_weights=True`` (default)
   - Int8 inference. Pass the argument ``has_fp16_weights=False``
4. To use the full LLM.int8() method, use the ``threshold=k`` argument. We recommend ``k=6.0``.
```python
# LLM.int8()
linear = bnb.nn.Linear8bitLt(dim1, dim2, bias=True, has_fp16_weights=False, threshold=6.0)
# inputs need to be fp16
out = linear(x.to(torch.float16))
```

### Using Int8 inference with HuggingFace Transformers:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
  'decapoda-research/llama-7b-hf,
  device_map='auto',
  load_in_8bit=True,
  max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB')
```
