# test_ner.py --model_file <path to the trained model> --char_embeddings [ 0 | 1 ] --layer_normalization [ 0 | 1 ] --crf [ 0 | 1 ] 
# --test_data_file <path to a file in the same format as original train file with random  NER / POS tags for each token> 
# --output_file <file in the same format as the test data file with random NER tags replaced with the predictions> 
# --glove_embeddings_file <path to file containing glove embeddings> --vocabulary_input_file <path to the vocabulary file written while training>

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--initialization', type=str)
parser.add_argument('--model_file', type=str)
parser.add_argument('--char_embeddings', type=int)
parser.add_argument('--layer_normalization', type=int)
parser.add_argument('--crf', type=int)
parser.add_argument('--test_data_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--glove_embeddings_file', type=str)
parser.add_argument('--vocabulary_input_file', type = str)
args=parser.parse_args()


#simple bilstm with random/glove
if(args.char_embeddings == 0 and args.layer_normalization == 0 and args.crf == 0):
    os.system('python3 TEST2_1_1_bilstm_random_glove.py --initialization {} --model_file {} --output_file {} --test_data_file {} --glove_embeddings_file {} --vocabulary_input_file {}'.format(args.initialization, args.model_file, args.output_file, args.test_data_file, args.glove_embeddings_file, args.vocabulary_input_file ))

#bilstm+char+glove/random
elif(args.char_embeddings == 1 and args.layer_normalization == 0 and args.crf == 0):
    os.system('python3 TEST2_1_3_bilstm_char_random_glove.py --initialization {} --model_file {} --output_file {} --test_data_file {} --glove_embeddings_file {} --vocabulary_input_file {}'.format(args.initialization, args.model_file, args.output_file, args.test_data_file, args.glove_embeddings_file, args.vocabulary_input_file ))

#bilstm+layernorm+char+glove/random
elif(args.char_embeddings == 1 and args.layer_normalization == 1 and args.crf == 0):
    os.system('python3 TEST2_1_4_bilstm_layernorm_char_random_glove.py --initialization {} --model_file {} --output_file {} --test_data_file {} --glove_embeddings_file {} --vocabulary_input_file {}'.format(args.initialization, args.model_file, args.output_file, args.test_data_file, args.glove_embeddings_file, args.vocabulary_input_file ))

#bilstm+glove/random+layernorm+char+crf
elif(args.char_embeddings == 1 and args.layer_normalization == 1 and args.crf == 1):
    os.system('python3 TEST2_2a_bilstm_crf_layernorm_char_random_glove.py --initialization {} --model_file {} --output_file {} --test_data_file {} --glove_embeddings_file {} --vocabulary_input_file {}'.format(args.initialization, args.model_file, args.output_file, args.test_data_file, args.glove_embeddings_file, args.vocabulary_input_file ))

elif(args.char_embeddings == 1):
    os.system('python3 TEST2_2a_bilstm_crf_layernorm_char_random_glove.py --initialization {} --model_file {} --output_file {} --test_data_file {} --glove_embeddings_file {} --vocabulary_input_file {}'.format(args.initialization, args.model_file, args.output_file, args.test_data_file, args.glove_embeddings_file, args.vocabulary_input_file ))
################################ extra

# #bilstm+glove/random+crf
# elif(args.char_embeddings == 0 and args.layer_normalization == 0 and args.crf == 1):
#     os.system('python3 TEST2_2c_bilstm_crf_random_glove.py --initialization {} --model_file {} --output_file {} --test_data_file {} --glove_embeddings_file {} --vocabulary_input_file {}'.format(args.initialization, args.model_file, args.output_file, args.test_data_file, args.glove_embeddings_file, args.vocabulary_input_file ))

# #bilstm+crf+char+glove
# elif(args.char_embeddings == 0 and args.layer_normalization == 0 and args.crf == 1):
#     os.system('python3 TEST2_2b_bilstm_crf_char_random_glove.py --initialization {} --model_file {} --output_file {} --test_data_file {} --glove_embeddings_file {} --vocabulary_input_file {}'.format(args.initialization, args.model_file, args.output_file, args.test_data_file, args.glove_embeddings_file, args.vocabulary_input_file ))
