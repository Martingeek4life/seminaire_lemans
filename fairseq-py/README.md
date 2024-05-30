# seminaire_lemans
Documentation de mes experimentations sur fairseq

## Generer les BPE
1-	Learn BPE

subword-nmt learn-bpe -s 4000 < new_testament_ew_train.txt > bpe_codes_ew

subword-nmt learn-bpe -s 4000 < new_testament_fr_train.txt > bpe_codes_fr

2-	Apply BPE

subword-nmt apply-bpe -c bpe_codes_ew < new_testament_ew_train.txt > train.bpe.ew

subword-nmt apply-bpe -c bpe_codes_fr < new_testament_fr_train.txt > train.bpe.fr

subword-nmt apply-bpe -c bpe_codes_ew < new_testament_ew_test.txt > test.bpe.ew

subword-nmt apply-bpe -c bpe_codes_ew < new_testament_ew_dev.txt > dev.bpe.ew

subword-nmt apply-bpe -c bpe_codes_fr < new_testament_fr_test.txt > test.bpe.fr

subword-nmt apply-bpe -c bpe_codes_fr < new_testament_fr_dev.txt > dev.bpe.fr

## Preprocess Data for Fairseq

Fairseq requires data to be in a binary format:

preprocess.py --source-lang ew --target-lang fr --trainpref train.bpe --validpref dev.bpe --testpref test.bpe --destdir data-bin

## Train the model with conv architecture good for low resources

python3 train.py data-bin/   --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 3000   --arch fconv_wmt_en_ro --save-dir checkpoints/ 

## Generate translation
python3 generate.py data-bin/ --path checkpoints/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe

i resolve issue ofconv_tbc like: return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding[0]) link help: https://github.com/EdinburghNLP/XSum/issues/11

i had be inspired on this link: https://github.com/wengong-jin/fairseq-py and https://github.com/sliedes/fairseq-py

