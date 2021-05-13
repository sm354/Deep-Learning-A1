# train_ner.py --initialization [random | glove ] --char_embeddings [ 0 | 1 ] --layer_normalization [ 0 | 1 ] --crf [ 0 | 1 ] 
# --output_file <path to the trained model> --data_dir <directory containing data> --glove_embeddings_file <path to file containing glove embeddings> 
# --vocabulary_output_file <path to the file in which vocabulary will be written>
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--initialization', type=str)
parser.add_argument('--char_embeddings', type=int)
parser.add_argument('--layer_normalization', type=int)
parser.add_argument('--crf', type=int)
parser.add_argument('--output_file', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--glove_embeddings_file', type=str)
parser.add_argument('--vocabulary_output_file', type = str)
args=parser.parse_args()

#simple bilstm with random/glove
if(args.char_embeddings == 0 and args.layer_normalization == 0 and args.crf == 0):
    os.system('python3 Train2_1_1_bilstm_random_glove.py --initialization {} --output_file {} --data_dir {} --glove_embeddings_file {} --vocabulary_output_file {}'.format(args.initialization, args.output_file, args.data_dir, args.glove_embeddings_file, args.vocabulary_output_file ))

#bilstm+char+glove/random
elif(args.char_embeddings == 1 and args.layer_normalization == 0 and args.crf == 0):
    os.system('python3 Train2_1_3_bilstm_char_random_glove.py --initialization {} --output_file {} --data_dir {} --glove_embeddings_file {} --vocabulary_output_file {}'.format(args.initialization, args.output_file, args.data_dir, args.glove_embeddings_file, args.vocabulary_output_file ))

#bilstm+layernorm+char+glove/random
elif(args.char_embeddings == 1 and args.layer_normalization == 1 and args.crf == 0):
    os.system('python3 Train2_1_4_bilstm_layernorm_char_random_glove.py --initialization {} --output_file {} --data_dir {} --glove_embeddings_file {} --vocabulary_output_file {}'.format(args.initialization, args.output_file, args.data_dir, args.glove_embeddings_file, args.vocabulary_output_file ))

#bilstm+glove/random+layernorm+char+crf
elif(args.char_embeddings == 1 and args.layer_normalization == 1 and args.crf == 1):
    os.system('python3 Train2_2a_bilstm_crf_layernorm_char_random_glove.py --initialization {} --output_file {} --data_dir {} --glove_embeddings_file {} --vocabulary_output_file {}'.format(args.initialization, args.output_file, args.data_dir, args.glove_embeddings_file, args.vocabulary_output_file ))

elif(args.char_embeddings == 1):
    os.system('python3 Train2_2a_bilstm_crf_layernorm_char_random_glove.py --initialization {} --output_file {} --data_dir {} --glove_embeddings_file {} --vocabulary_output_file {}'.format(args.initialization, args.output_file, args.data_dir, args.glove_embeddings_file, args.vocabulary_output_file ))


################################ extra

# #bilstm+glove/random+crf
# elif(args.char_embeddings == 0 and args.layer_normalization == 0 and args.crf == 1):
#     os.system('python3 2_2c_bilstm_crf_random_glove.py --initialization {} --output_file {} --data_dir {} --glove_embeddings_file {} --vocabulary_output_file {}'.format(args.initialization, args.output_file, args.data_dir, args.glove_embeddings_file, args.vocabulary_output_file ))

# #bilstm+crf+char+glove
# elif(args.char_embeddings == 0 and args.layer_normalization == 0 and args.crf == 1):
#     os.system('python3 2_2b_bilstm_crf_char_random_glove.py --initialization {} --output_file {} --data_dir {} --glove_embeddings_file {} --vocabulary_output_file {}'.format(args.initialization, args.output_file, args.data_dir, args.glove_embeddings_file, args.vocabulary_output_file ))
