wget https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/iwslt2015-zhen.tar.gz
mkdir -p data/iwslt
tar -xvzf iwslt2015-zhen.tar.gz -C data/
rm iwslt2015-zhen.tar.gz
