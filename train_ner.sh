#!/bin/sh
### Set the job name (for your reference)
#PBS -N ner1
### Set the project name, your department code by default
#PBS -P ee
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ngpus=1

### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=6:00:00

#PBS -l software=python
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

module purge
module load apps/anaconda/3

# python3 train_ner.py --initialization random --char_embeddings 0 --layer_normalization 0 --crf 0 --output_file tempfiles/1.pth --data_dir ../NER_Dataset/ner-gmb --glove_embeddings_file kachra --vocabulary_output_file tempfiles/1.vocab
# python3 train_ner.py --initialization random --char_embeddings 1 --layer_normalization 0 --crf 0 --output_file tempfiles/2.pth --data_dir ../NER_Dataset/ner-gmb --glove_embeddings_file kachra --vocabulary_output_file tempfiles/2.vocab
# python3 train_ner.py --initialization random --char_embeddings 1 --layer_normalization 1 --crf 0 --output_file tempfiles/3.pth --data_dir ../NER_Dataset/ner-gmb --glove_embeddings_file kachra --vocabulary_output_file tempfiles/3.vocab
# python3 train_ner.py --initialization random --char_embeddings 1 --layer_normalization 1 --crf 1 --output_file tempfiles/4.pth --data_dir ../NER_Dataset/ner-gmb --glove_embeddings_file kachra --vocabulary_output_file tempfiles/4.vocab

python3 train_ner.py --initialization random --char_embeddings 0 --layer_normalization 0 --crf 0 --output_file useful_data/1_epoch.pth --data_dir useful_data/ --glove_embeddings_file useful_data/glove.6B.100d.txt --vocabulary_output_file useful_data/1_0epochs.vocab
# python3 train_ner.py --initialization glove --char_embeddings 1 --layer_normalization 0 --crf 0 --output_file tempfiles/2_g.pth --data_dir ../NER_Dataset/ner-gmb --glove_embeddings_file ../glove_data/glove.6B.100d.txt --vocabulary_output_file tempfiles/2_g.vocab
# python3 train_ner.py --initialization glove --char_embeddings 1 --layer_normalization 1 --crf 0 --output_file tempfiles/3_g.pth --data_dir ../NER_Dataset/ner-gmb --glove_embeddings_file ../glove_data/glove.6B.100d.txt --vocabulary_output_file tempfiles/3_g.vocab
# python3 train_ner.py --initializßation glove --char_embeddings 1 --layer_normalization 1 --crf 1 --output_file tempfiles/4_g.pth --data_dir ../NER_Dataset/ner-gmb --glove_embeddings_file ../glove_data/glove.6B.100d.txt --vocabulary_output_file tempfiles/4_g.vocab


#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE