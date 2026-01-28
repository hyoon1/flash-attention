#!/bin/bash

TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 FLASH_ATTN_LOG_DISABLE=1 python measure_flash_attn_tflops.py --sdpa --plot

## gfx11
#TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 FLASH_ATTN_LOG_DISABLE=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python measure_flash_attn_tflops.py --backend triton --sdpa --plot

