#!/bin/bash
# Cleaned-up EN→DE preprocessing script
# Based on original WMT17 pipeline with BPE & Moses tokenization

echo "Cloning Moses (tokenization)..."
[ ! -d mosesdecoder ] && git clone https://github.com/moses-smt/mosesdecoder.git

echo "Cloning Subword NMT (optional)..."
[ ! -d subword-nmt ] && git clone https://github.com/rsennrich/subword-nmt.git

# Paths
SCRIPTS=/hdd2/giri/ContinualMT/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

# FastBPE + pretrained WMT19 EN–DE codes
FASTBPE=/hdd2/giri/ContinualMT/fastBPE
BPECODES=/hdd2/giri/ContinualMT/pretrained_models/wmt19.de-en.joined-dict.ensemble/bpecodes
VOCAB=/hdd2/giri/ContinualMT/pretrained_models/wmt19.de-en.joined-dict.ensemble/dict.de.txt  # target = de

# Download URLs
URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)

FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v12.tgz"
    "dev.tgz"
    "test-full.tgz"
)

CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training/news-commentary-v12.de-en"
)

# OUTDIR
OUTDIR=/hdd2/giri/ContinualMT/wmt17_en_de

if [ ! -d "$SCRIPTS" ]; then
    echo "Error: Moses scripts not found at $SCRIPTS"
    exit 1
fi

# Language direction
src=en
tgt=de
lang=en-de
prep=$OUTDIR
tmp=$prep/tmp
orig=orig
dev=dev/newstest2016

mkdir -p $orig $tmp $prep
cd $orig

### --- DOWNLOAD DATA --- ###
for i in "${!URLS[@]}"; do
    file=${FILES[i]}
    url=${URLS[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping."
    else
        wget $url
        if [ -f $file ]; then
            echo "Downloaded $file"
            tar -xf $file
        else
            echo "Failed to download $file"
            exit 1
        fi
    fi
done
cd ..

### --- PREPROCESS TRAINING DATA --- ###
echo "Pre-processing train data..."
for l in $src $tgt; do
    rm -f $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l \
            >> $tmp/train.tags.$lang.tok.$l
    done
done

### --- PREPROCESS TEST DATA --- ###
echo "Pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi

    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
        perl $TOKENIZER -threads 8 -a -l $l \
        > $tmp/test.$l
done

### --- SPLIT TRAIN/VALID --- ###
echo "Splitting train & valid..."
for l in $src $tgt; do
    awk '{if (NR % 100 == 0) print $0;}'  $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR % 100 != 0) print $0;}'  $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done

### --- APPLY BPE --- ###
echo "Applying BPE..."
for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "Applying BPE to $f..."
        $FASTBPE/fast applybpe $tmp/bpe.$f $tmp/$f $BPECODES $VOCAB
    done
done

### --- CLEAN --- ###
echo "Cleaning..."
perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

### --- COPY TEST --- ###
for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done

echo "DONE. Preprocessed EN→DE data is in: $OUTDIR"
