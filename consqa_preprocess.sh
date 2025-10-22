src='en'
tgt='de'
TEXT=data/consqa-m30k-en2$tgt

CUDA_VISIBLE_DEVICES=1,2,4 fairseq-preprocess --source-lang $src --target-lang $tgt \
 --trainpref $TEXT/train \
 --validpref $TEXT/valid \
 --testpref $TEXT/test.2016,$TEXT/test.2017,$TEXT/test.coco,$TEXT/test.2018 \
 --destdir data-bin/multi30k.en-$tgt.consqa_mmt\
 --workers 4 --joined-dictionary \
 --srcquery $TEXT/train_$src --srcans $TEXT/train_$src \
 --srcgptquery $TEXT/gpt_train_$src --srcgptans $TEXT/gpt_train_$src \
 --task consqa_mmt \
 --tgtquery $TEXT/train_$tgt --tgtans $TEXT/train_$tgt \