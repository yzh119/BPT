mkdir -p data/text8/text8
cd data/text8/text8
wget --continue http://mattmahoney.net/dc/text8.zip
wget https://raw.githubusercontent.com/kimiyoung/transformer-xl/master/prep_text8.py
python prep_text8.py
cd ../../..
