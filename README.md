# seminaire_lemans
Documentation de mes experimentations sur fairseq

I-	Generer les BPE

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
II-	Preprocess Data for Fairseq
Fairseq requires data to be in a binary format:
fairseq-preprocess --source-lang src --target-lang tgt \ --trainpref train.bpe --validpref valid.bpe  --testpref test.bpe \  --destdir data-bin --thresholdtgt 0 --thresholdsrc 0 \  --workers 4

