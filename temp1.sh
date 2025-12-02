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




