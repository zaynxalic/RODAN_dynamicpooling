# RODAN
A fully convolutional architecture for basecalling nanopore RNA sequencing data

Generated Taiyaki RNA data: https://doi.org/10.5281/zenodo.4556884

RNA training and validation data: https://doi.org/10.5281/zenodo.4556950

RNA test data: https://doi.org/10.5281/zenodo.4557004

## Requirements
* Python 3
* torch >= 1.4.0 <= 1.8.0
* numpy
* h5py
* ont-fast5-api
* fast-ctc-decode
* pyyaml
* tensorboard
* pytorch-ranger (only for training)

## Installation

Create a python virtual environment. 
```
python3 -m venv virtualenv
source virtualenv/bin/activate
git clone https://github.com/biodlab/RODAN.git
cd RODAN
pip install -r requirements.txt
```

## Training

To train, download the RNA training data from the above link.

```
mkdir runs
pip install pytorch-ranger
./model_dp.py -c rna.config -n NAME -l or 

python model_py.py -c rna.config -n runs -l > output.txt
```


## Basecalling

To basecall (must be run from root directory):

`./basecall.py /path/to/fast5files > outfile.fasta` or 

`python basecall.py /path/to/fast5files > outfile.fasta`


## Minimap matching
`minimap2 --secondary=no -ax map-ont -t 32 --cs genomefile fastafile > file.sam`


## Accuracy 
`python accuracy.py  /path/to/samfile /path/to/genomefile`


Basecall will recursively search in the specified directory for all fast5 files which can be single or multi fast5 files.

### Test data
Five samples of human RNA fast5 data is provided in test-data.tgz.

