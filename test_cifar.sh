#!/bin/sh
### Set the job name (for your reference)
#PBS -N f
### Set the project name, your department code by default
#PBS -P ee
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ngpus=1

### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=00:30:00

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

# python test_cifar.py --model_file ./pretrained_models/part_1.1.pth --normalization torch_bn --n 2 --test_data_file ./sample_test_data/cifar_test.csv --output_file ./saving_results/1.1_test_out.csv
# python test_cifar.py --model_file ./pretrained_models/part_1.2_bin.pth --normalization bin --n 2 --test_data_file ./sample_test_data/cifar_test.csv --output_file ./saving_results/1.2_bin_test_out.csv
# python test_cifar.py --model_file ./pretrained_models/part_1.2_bn.pth --normalization bn --n 2 --test_data_file ./sample_test_data/cifar_test.csv --output_file ./saving_results/1.2_bn_test_out.csv
# python test_cifar.py --model_file ./pretrained_models/part_1.2_gn.pth --normalization gn --n 2 --test_data_file ./sample_test_data/cifar_test.csv --output_file ./saving_results/1.2_gn_test_out.csv
# python test_cifar.py --model_file ./pretrained_models/part_1.2_in.pth --normalization in --n 2 --test_data_file ./sample_test_data/cifar_test.csv --output_file ./saving_results/1.2_in_test_out.csv
# python test_cifar.py --model_file ./pretrained_models/part_1.2_ln.pth --normalization ln --n 2 --test_data_file ./sample_test_data/cifar_test.csv --output_file ./saving_results/1.2_ln_test_out.csv
# python test_cifar.py --model_file ./pretrained_models/part_1.2_nn.pth --normalization nn --n 2 --test_data_file ./sample_test_data/cifar_test.csv --output_file ./saving_results/1.2_nn_test_out.csv

#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE
