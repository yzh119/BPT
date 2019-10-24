mkdir -p data/enwik8/enwik8
cd data/enwik8/enwik8
wget --continue http://mattmahoney.net/dc/enwik8.zip
wget https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
python prep_enwik8.py
cd ../../..
