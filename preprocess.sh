src='en'
tgt='de'
TEXT=data/multi30k-en2$tgt

fairseq-preprocess --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test.2016,$TEXT/test.2017,$TEXT/test.coco,$TEXT/test.2018 \
  --destdir data-bin/multi30k.en-de.mmt_vqa_2\
  --workers 4 --joined-dictionary \
  --srcquery $TEXT/train_$src --srcans $TEXT/train_$src \
  --srcdict data/dict_vqa.en2de.txt \
  --task mmt_vqa_2 \
  --tgtquery $TEXT/train_$tgt --tgtans $TEXT/train_$tgt \

#srcquery:train.query_en
#tgtquery:train.query_de

# python get_image_name.py