# Generic domain (43.9M) - combination of multiple corpora
opus_get -s de -t en -d Europarl
opus_get -s de -t en -d News-Commentary
opus_get -s de -t en -d EUbookshop

# Software domain (223K)
opus_get -s de -t en -d GNOME
opus_get -s de -t en -d KDE4
opus_get -s de -t en -d Ubuntu

# Koran domain (18K)
opus_get -s de -t en -d Tanzil

# Law domain (467K)
opus_get -s de -t en -d JRC-Acquis
opus_get -s de -t en -d DGT

# Medical domain (248K)
opus_get -s de -t en -d EMEA

# Subtitles domain (500K)
opus_get -s de -t en -d OpenSubtitles


 export WORKDIR_ROOT=/hdd2/giri/ContinualMT
 export SPM_PATH=/hdd2/giri/ContinualMT/SPM_PATH
export SCRIPTS=/hdd2/giri/ContinualMT/mosesdecoder

python ../../fairseq_cli/preprocess.py     --source-lang de     --target-lang en     --trainpref train     --validpref dev     --testpref test     --destdir /hdd2/giri/ContinualMT/data-bin/it     --workers 60  

bash cl_scripts/fmalloc/fmalloc_train.sh > /hdd2/giri/ContinualMT/logs/run_$(date +%Y-%m-%d_%H-%M-%S).out 2> /hdd2/giri/ContinualMT/logs/run_$(date +%Y-%m-%d_%H-%M-%S).err


#!/bin/bash

# Create directories
mkdir -p wmt19_de_en wmt20_de_en wmt21_de_en

# Download test sets (easiest to get)
echo "Downloading test sets..."

# WMT19
sacrebleu -t wmt17 -l de-en --echo src > wmt17_de_en/train.de
sacrebleu -t wmt17 -l de-en --echo ref > wmt17_de_en/train.en

# WMT20
sacrebleu -t wmt20 -l de-en --echo src > wmt20_de_en/train.de
sacrebleu -t wmt20 -l de-en --echo ref > wmt20_de_en/train.en

# WMT21
sacrebleu -t wmt21 -l de-en --echo src > wmt21_de_en/train.de
sacrebleu -t wmt21 -l de-en --echo ref > wmt21_de_en/train.en


# Create directory
mkdir -p wmt17_de_en

# Training data (WMT17 training corpus)
sacrebleu -t wmt17 -l de-en --echo src > wmt17_de_en/train.de
sacrebleu -t wmt17 -l de-en --echo ref > wmt17_de_en/train.en

# Validation data (WMT16 test set)
sacrebleu -t wmt16 -l de-en --echo src > wmt17_de_en/valid.de
sacrebleu -t wmt16 -l de-en --echo ref > wmt17_de_en/valid.en

# Test data (WMT17 test set)
sacrebleu -t wmt17 -l de-en --echo src > wmt17_de_en/test.de
sacrebleu -t wmt17 -l de-en --echo ref > wmt17_de_en/test.en

echo "Test sets downloaded. For training data, you may need to:"
echo "1. Check the ContinualMT repository for their preprocessed data"
echo "2. Download from official WMT websites and preprocess yourself"
echo "3. Use Hugging Face datasets for easier access"
