#! /usr/bin/bash
set -e

export PYTHONPATH=$(pwd):$PYTHONPATH

batch_size=128
beam=5
src_lang=en
tgt_lang=fr
ensemble=10
data_dir=multi30k.en-fr.consqa_mmt

gpu=0,1,2,3,4,5,6,7
length_penalty=1.0
arch=transformer_consqa_mmt

tag=en-fr-consqa_mmt_0.001_2048_0.2w1
vision_model_name=vit-mae-base
use_vlm_text_encoder=0

model_dir=checkpoints/$tag

# Ensemble 10 checkpoints
if [ ! -e "$model_dir/last${ensemble}.ensemble.pt" ]; then
    python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last${ensemble}.ensemble.pt --num-epoch-checkpoints $ensemble
fi
checkpoint=last${ensemble}.ensemble.pt

export CUDA_VISIBLE_DEVICES=$gpu

echo "=== Testing with ensemble ${ensemble} ==="
for who in test test1 test2; do
  output=$model_dir/translation_ensemble_$who.log

  cmd="python3 -m fairseq_cli.generate data-bin/$data_dir
    -s $src_lang -t $tgt_lang
    --task consqa_mmt
    --path $model_dir/$checkpoint
    --gen-subset $who
    --batch-size $batch_size --beam $beam --lenpen $length_penalty
    --quiet --remove-bpe --scoring sacrebleu
    --output $model_dir/hyp_ensemble_$who.txt
    --image-name-dir data-bin/$data_dir
    --ptm-name $vision_model_name
    --source-sentence-dir data-bin"

  if [ $use_vlm_text_encoder -eq 1 ]; then
    cmd=${cmd}" --use-vlm-text-encoder "
  fi

  cmd=${cmd}" | tee "${output}
  eval $cmd

  python3 rerank.py $model_dir/hyp_ensemble_$who.txt $model_dir/hyp_ensemble_$who.sorted
  echo "[$who] done"
done