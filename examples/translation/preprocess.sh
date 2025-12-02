
DATA=5
TEXT=/hdd2/giri/ContinualMT/opus_data/subtitles
python fairseq_cli/preprocess.py --source-lang de --target-lang en \
    --trainpref $TEXT/train.bpe --validpref $TEXT/dev.bpe --testpref $TEXT/test.bpe \
    --destdir data-bin/$DATA \
    --srcdict pretrained_models/wmt19.de-en.joined-dict.ensemble/dict.de.txt \
    --tgtdict pretrained_models/wmt19.de-en.joined-dict.ensemble/dict.en.txt \
    --workers 20