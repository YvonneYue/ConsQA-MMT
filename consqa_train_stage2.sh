#! /usr/bin/bash
#set -e
#
#SA_attention_dropout=0.1
#SA_image_dropout=0.1
#SA_text_dropout=0
#vqa_SA_attention_dropout=0.1
#vqa_SA_image_dropout=0.1
#vqa_SA_text_dropout=0
#
#device=1,2
#gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`
#gpu_num=2
#
#src_lang=en
#tgt_lang=de
#vision_model_name=facebook/vit-mae-base
#use_vlm_text_encoder=0
#
#
#keep_last_epochs=100
#patience=10
#criterion=consqa_mmt_label_smoothed_cross_entropy
#
#
#data_dir=pretrain.en-de.consqa_mmt
#arch=transformer_consqa_mmt
#
#fp16=1
#lr=0.001
#warmup=4000
#max_tokens=2048  #就是batch_size
#update_freq=1
#dropout=0.3
#weight1=0.2
#weitht2=0.1
#tag=1225_en-de_2
#save_dir=checkpoints/$tag
#
#if [ ! -d $save_dir ]; then
#        mkdir -p $save_dir
#fi
#
#cp ${BASH_SOURCE[0]} $save_dir/train.sh
#
#cmd="fairseq-train data-bin/$data_dir
#  --save-dir $save_dir
#  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
#  --arch $arch
#  --dropout $dropout
#  --criterion $criterion --label-smoothing 0.1
#  --task consqa_mmt
#  --optimizer adam --adam-betas '(0.9, 0.98)'
#  --lr $lr --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
#  --max-tokens $max_tokens
#  --update-freq $update_freq
#  --share-all-embeddings
#  --find-unused-parameters
#  --skip-invalid-size-inputs-valid-test
#  --patience $patience
#  --keep-last-epochs $keep_last_epochs
#  --image-name-dir data-bin/$data_dir
#  --ptm-name $vision_model_name
#  --vision-model $vision_model_name
#  --weight1 $weight1
#  --weight2 $weitht2
#  --source-sentence-dir data-bin"
#
#if [ $use_vlm_text_encoder -eq 1 ]; then
#cmd=${cmd}" --use-vlm-text-encoder "
#fi
#if [ $fp16 -eq 1 ]; then
#cmd=${cmd}" --fp16 "
#fi
#if [ -n "$SA_image_dropout" ]; then
#cmd=${cmd}" --SA-image-dropout "${SA_image_dropout}
#fi
#if [ -n "$SA_text_dropout" ]; then
#cmd=${cmd}" --SA-text-dropout "${SA_text_dropout}
#fi
#if [ -n "$SA_attention_dropout" ]; then
#cmd=${cmd}" --SA-attention-dropout "${SA_attention_dropout}
#fi
#if [ -n "$vqa_SA_image_dropout" ]; then
#cmd=${cmd}" --vqa-SA-image-dropout "${vqa_SA_image_dropout}
#fi
#if [ -n "$vqa_SA_text_dropout" ]; then
#cmd=${cmd}" --vqa-SA-text-dropout "${vqa_SA_text_dropout}
#fi
#if [ -n "$vqa_SA_attention_dropout" ]; then
#cmd=${cmd}" --vqa-SA-attention-dropout "${vqa_SA_attention_dropout}
#fi
#
#export CUDA_VISIBLE_DEVICES=$device
#
#if [ -f $save_dir/train.log ]; then
#    echo "File $save_dir/train.log exists!"
#    exit
#fi
#
#cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
#eval $cmd
#tail -f $save_dir/train.log


#! /usr/bin/bash
#################stage 2:
set -e

SA_attention_dropout=0.1
SA_image_dropout=0.1
SA_text_dropout=0
vqa_SA_attention_dropout=0.1
vqa_SA_image_dropout=0.1
vqa_SA_text_dropout=0

device=0,1,2,4
gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`
gpu_num=4

src_lang=en
tgt_lang=de
vision_model_name=facebook/vit-mae-base
use_vlm_text_encoder=0


keep_last_epochs=100
criterion=consqa_mmt_label_smoothed_cross_entropy

data_dir=pretrain.en-de.consqa_mmt
#arch=transformer_consqa_mmt
arch=transformer_consqa_mmt_2sa_2decoder


fp16=1
lr=0.005
warmup=4000
max_tokens=1024  #就是batch_size
update_freq=1
dropout=0.3
weight1=0.2
weitht2=0.1
tag=pretrain_en-de_0.005lr_0.2w1_0.1w2
save_dir=checkpoints/$tag
pretrain=0

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi

cp ${BASH_SOURCE[0]} $save_dir/train_stage2_2.sh

cmd="fairseq-train data-bin/$data_dir
  --save-dir $save_dir
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --dropout $dropout
  --criterion $criterion --label-smoothing 0.1
  --task consqa_mmt
  --optimizer adam --adam-betas '(0.9, 0.98)'
  --lr $lr --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --max-tokens $max_tokens
  --update-freq $update_freq
  --share-all-embeddings
  --find-unused-parameters
  --skip-invalid-size-inputs-valid-test
  --patience 10
  --keep-last-epochs $keep_last_epochs
  --image-name-dir data-bin/$data_dir
  --ptm-name $vision_model_name
  --vision-model $vision_model_name
  --pretrain $pretrain
  --source-sentence-dir data-bin
  --num-workers 0
  --restore-file checkpoint45_stage2.pt
  --checkpoint-suffix _stage2
  --weight1 $weight1
  --weight2 $weitht2"

if [ $pretrain -eq 1 ]; then
cmd=${cmd}" --pretrain True"
fi
if [ $pretrain -eq 0 ]; then
cmd=${cmd}" --pretrain False"
fi
if [ $use_vlm_text_encoder -eq 1 ]; then
cmd=${cmd}" --use-vlm-text-encoder "
fi
if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi
if [ -n "$SA_image_dropout" ]; then
cmd=${cmd}" --SA-image-dropout "${SA_image_dropout}
fi
if [ -n "$SA_text_dropout" ]; then
cmd=${cmd}" --SA-text-dropout "${SA_text_dropout}
fi
if [ -n "$SA_attention_dropout" ]; then
cmd=${cmd}" --SA-attention-dropout "${SA_attention_dropout}
fi
if [ -n "$vqa_SA_image_dropout" ]; then
cmd=${cmd}" --vqa-SA-image-dropout "${vqa_SA_image_dropout}
fi
if [ -n "$vqa_SA_text_dropout" ]; then
cmd=${cmd}" --vqa-SA-text-dropout "${vqa_SA_text_dropout}
fi
if [ -n "$vqa_SA_attention_dropout" ]; then
cmd=${cmd}" --vqa-SA-attention-dropout "${vqa_SA_attention_dropout}
fi
if [ -n "$checkpoint_suffix" ]; then
cmd=${cmd}" --checkpoint-suffix "${checkpoint_suffix}
fi

export CUDA_VISIBLE_DEVICES=$device

if [ -f $save_dir/train_stage2_2.log ]; then
    echo "File $save_dir/train_stage2_2.log exists!"
    exit
fi

cmd="nohup "${cmd}" > $save_dir/train_stage2_2.log 2>&1 &"
eval $cmd
tail -f $save_dir/train_stage2_2.log