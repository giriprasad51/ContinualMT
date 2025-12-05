
DATA=wmt17_de_en
TEXT=/hdd2/giri/ContinualMT/data-bin1/$DATA
python fairseq_cli/preprocess.py --source-lang de --target-lang en \
    --trainpref $TEXT/train.bpe --validpref $TEXT/dev.bpe --testpref $TEXT/test.bpe \
    --destdir /hdd2/giri/ContinualMT/data-bin/$DATA \
    --srcdict /hdd2/giri/ContinualMT/pretrained_models/wmt19.de-en.joined-dict.ensemble/dict.de.txt \
    --tgtdict /hdd2/giri/ContinualMT/pretrained_models/wmt19.de-en.joined-dict.ensemble/dict.en.txt \
    --workers 20 

# python ../../fairseq_cli/preprocess.py     --source-lang de     --target-lang en     --trainpref train     --validpref dev     --testpref test     --destdir /hdd2/giri/ContinualMT/data-bin/it     --workers 60  