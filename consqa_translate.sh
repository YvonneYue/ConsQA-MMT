#! /usr/bin/bash
set -e

batch_size=128
beam=5
src_lang=en
tgt_lang=de
ensemble=10
data_dir=multi30k.en-de.consqa_mmt

gpu=1,2,4
length_penalty=1.0
arch=transformer_consqa_mmt

tag=en-de_0.001lr_0.2w1_0.1w2
vision_model_name=facebook/vit-mae-base
use_vlm_text_encoder=0

model_dir=checkpoints/$tag
checkpoint=checkpoint_last.pt
cp ${BASH_SOURCE[0]} $model_dir/translate.sh

if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi

export CUDA_VISIBLE_DEVICES=$gpu

for who in test test1 test2 test3; do
  output=$model_dir/translation_$who.log

  cmd="fairseq-generate data-bin/$data_dir
    -s $src_lang -t $tgt_lang --task mmt+nmt_vqa_2
    --path $model_dir/$checkpoint
    --gen-subset $who
    --batch-size $batch_size --beam $beam --lenpen $length_penalty
    --quiet --remove-bpe
    --output $model_dir/hyp_$who.txt
    --image-name-dir data-bin/$data_dir
    --ptm-name $vision_model_name
    --source-sentence-dir data-bin
    --arch $arch"

  if [ $use_vlm_text_encoder -eq 1 ]; then
  cmd=${cmd}" --use-vlm-text-encoder "
  fi

  cmd=${cmd}" | tee "${output}
  eval $cmd

  python3 rerank.py $model_dir/hyp_$who.txt $model_dir/hyp_$who.sorted

  task=multi30k-$src_lang-$tgt_lang

  if [ "$task" == "multi30k-en-de" ] && [ $who == "test" ]; then
    ref=data/multi30k/test.2016.de
  elif [ "$task" == "multi30k-en-de" ] && [ $who == "test1" ]; then
 	  ref=data/multi30k/test.2017.de
  elif [ "$task" == "multi30k-en-de" ] && [ $who == "test2" ]; then
    ref=data/multi30k/test.coco.de
  elif [ "$task" == "multi30k-en-de" ] && [ $who == "test3" ]; then
    ref=data/multi30k/test.2018.de

  elif [ $task == "multi30k-en-fr" ] && [ $who == 'test' ]; then
    ref=data/multi30k/test.2016.fr
  elif [ $task == "multi30k-en-fr" ] && [ $who == 'test1' ]; then
    ref=data/multi30k/test.2017.fr
  elif [ $task == "multi30k-en-fr" ] && [ $who == 'test2' ]; then
    ref=data/multi30k/test.coco.fr
 elif [ "$task" == "multi30k-en-fr" ] && [ $who == "test3" ]; then
    ref=data/multi30k/test.2018.fr

  elif [ $task == "multi30k-en-cs" ] && [ $who == 'test' ]; then
    ref=data/multi30k/test.2016.cs
  elif [ $task == "multi30k-en-cs" ] && [ $who == 'test3' ]; then
    ref=data/multi30k/test.2018.cs
  fi

  hyp=$model_dir/hyp_$who.sorted
  python3 meteor.py $hyp $ref > $model_dir/meteor_$who.log
  cat $model_dir/meteor_$who.log

done