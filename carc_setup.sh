#!/bin/sh

module load gcc
module load vim
module load git
source env/bin/activate
export HF_HOME=/scratch2/madefran/hf_cache
export HF_DATASETS_CACHE="/scratch2/madefran/hf_cache"
export TRANSFORMERS_CACHE="/scratch2/madefran/hf_cache"
