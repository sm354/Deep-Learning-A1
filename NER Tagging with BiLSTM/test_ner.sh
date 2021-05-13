#!/bin/sh
### Set the job name (for your reference)
#PBS -N ner1_test
### Set the project name, your department code by default
#PBS -P ee
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ngpus=1

### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=02:00:00

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

# python3 test_ner.py --model_file tempfiles/1.pth --char_embeddings 0 --layer_normalization 0 --crf 0 --test_data_file ../submission_test_files/ner_test_file.txt --output_file tempfiles/1_test.txt --glove_embeddings_file kachra --vocabulary_input_file tempfiles/1.vocab
# python3 test_ner.py --model_file tempfiles/2.pth --char_embeddings 1 --layer_normalization 0 --crf 0 --test_data_file ../submission_test_files/ner_test_file.txt --output_file tempfiles/2_test.txt --glove_embeddings_file kachra --vocabulary_input_file tempfiles/2.vocab
# python3 test_ner.py --model_file tempfiles/3.pth --char_embeddings 1 --layer_normalization 1 --crf 0 --test_data_file ../submission_test_files/ner_test_file.txt --output_file tempfiles/3_test.txt --glove_embeddings_file kachra --vocabulary_input_file tempfiles/3.vocab
# python3 test_ner.py --model_file tempfiles/4.pth --char_embeddings 1 --layer_normalization 1 --crf 1 --test_data_file ../submission_test_files/ner_test_file.txt --output_file tempfiles/4_test.txt --glove_embeddings_file kachra --vocabulary_input_file tempfiles/4.vocab

#python3 test_ner.py --model_file tempfiles/1_g.pth --char_embeddings 0 --layer_normalization 0 --crf 0 --test_data_file ../submission_test_files/ner_test_file.txt --output_file tempfiles/1_g_test.txt --glove_embeddings_file ../glove_data/glove.6B.100d.txt --vocabulary_input_file tempfiles/1_g.vocab
# python3 test_ner.py --model_file tempfiles/2_g.pth --char_embeddings 1 --layer_normalization 0 --crf 0 --test_data_file ../submission_test_files/ner_test_file.txt --output_file tempfiles/2_g_test.txt --glove_embeddings_file ../glove_data/glove.6B.100d.txt --vocabulary_input_file tempfiles/2_g.vocab
# python3 test_ner.py --model_file tempfiles/3_g.pth --char_embeddings 1 --layer_normalization 1 --crf 0 --test_data_file ../submission_test_files/ner_test_file.txt --output_file tempfiles/3_g_test.txt --glove_embeddings_file ../glove_data/glove.6B.100d.txt --vocabulary_input_file tempfiles/3_g.vocab
# python3 test_ner.py --model_file tempfiles/4_g.pth --char_embeddings 1 --layer_normalization 1 --crf 1 --test_data_file ../submission_test_files/ner_test_file.txt --output_file tempfiles/4_g_test.txt --glove_embeddings_file ../glove_data/glove.6B.100d.txt --vocabulary_input_file tempfiles/4_g.vocab
python3 test_ner.py --initialization random --char_embeddings 0 --layer_normalization 0 --crf 0 --model_file useful_data/1.pth --test_data_file useful_data/test.txt --output_file useful_data/1.txt --glove_embeddings_file useful_data/glove.6B.100d.txt --vocabulary_input_file useful_data/1.vocab

#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE
