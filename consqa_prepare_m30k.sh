echo 'Cloning Moses github repository...'
git clone https://github.com/moses-smt/mosesdecoder.git

# tokenize and bpe for multi30k_vqa data
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

data_dir="data/consqa-M30k-VQA-de"
mkdir -p $data_dir/clean
mkdir -p $data_dir/clean_bpe

src="query"
tgt="ans"
lang=de

data_dir="data/consqa-M30k-VQA-de"

mkdir -p $data_dir/clean
mkdir -p $data_dir/clean_bpe

for l in $src $tgt; do
   cat $data_dir/gpt_train_en.$l | \
       perl $NORM_PUNC $l | \
       perl $REM_NON_PRINT_CHAR | \
       perl $LC | \
       perl $TOKENIZER -threads 8 -l $lang >> $data_dir/clean/gpt_train_en.$l
done

for l in $src $tgt; do
   cat $data_dir/train_en.$l | \
       perl $NORM_PUNC $l | \
       perl $REM_NON_PRINT_CHAR | \
       perl $LC | \
       perl $TOKENIZER -threads 8 -l $lang >> $data_dir/clean/train_en.$l
done

for l in $src $tgt; do
   cat $data_dir/train_$lang.$l | \
       perl $NORM_PUNC $l | \
       perl $REM_NON_PRINT_CHAR | \
       perl $LC | \
       perl $TOKENIZER -threads 8 -l $lang >> $data_dir/clean/train_$lang.$l
done

subword-nmt apply-bpe -c data/consqa-m30k-en2$lang/code < $data_dir/clean/gpt_train_en.query > $data_dir/clean_bpe/gpt_train_en.query
subword-nmt apply-bpe -c data/consqa-m30k-en2$lang/code < $data_dir/clean/gpt_train_en.ans > $data_dir/clean_bpe/gpt_train_en.ans
subword-nmt apply-bpe -c data/consqa-m30k-en2$lang/code < $data_dir/clean/train_en.query > $data_dir/clean_bpe/train_en.query
subword-nmt apply-bpe -c data/consqa-m30k-en2$lang/code < $data_dir/clean/train_en.ans > $data_dir/clean_bpe/train_en.ans
subword-nmt apply-bpe -c data/consqa-m30k-en2$lang/code < $data_dir/clean/train_$lang.query > $data_dir/clean_bpe/train_$lang.query
subword-nmt apply-bpe -c data/consqa-m30k-en2$lang/code < $data_dir/clean/train_$lang.ans > $data_dir/clean_bpe/train_$lang.ans


for l in $src $tgt; do
   cp $data_dir/clean_bpe/gpt_train_en.$l data/consqa-m30k-en2$lang/
done

for l in $src $tgt; do
   cp $data_dir/clean_bpe/train_en.$l data/consqa-m30k-en2$lang/
done

for l in $src $tgt; do
   cp $data_dir/clean_bpe/train_$lang.$l data/consqa-m30k-en2$lang/
done