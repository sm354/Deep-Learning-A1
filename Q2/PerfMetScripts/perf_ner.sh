#!/bin/sh
### Set the job name (for your reference)
#PBS -perf_test
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

python3 Perf2_1_1.py --model_file ../tempfiles/1_g.pth  --rootpath Results --Expname Simple_Glove --train_data_file ../../NER_Dataset/ner-gmb/train.txt --val_data_file ../../NER_Dataset/ner-gmb/dev.txt --test_data_file ../../NER_Dataset/ner-gmb/test.txt --vocabulary_input_file ../tempfiles/1_g.vocab

#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE