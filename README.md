# COL870 Deep Learning : Assignment 1

In this assignment, we do experimental work to report

1. the impact of different normalization schemes like batch norm, instance norm, group norm, etc in deep CNN models (using CIFAR-10).
2. the gain in performance when going from a simple Bi-LSTM (without pretrained word embeddings) to a Bi-LSTM with CRF layer. 

The problem statement is given [here](https://github.com/sm354/COL870-Assignment-1/blob/main/dl2021ass1.pdf) and detailed report can be found [here](https://github.com/sm354/COL870-Assignment-1/blob/main/Report.pdf). 

## ResNet and various Normalization Schemes

We use ResNet model [[He et al., 2016](https://arxiv.org/abs/1512.03385)] to solve image classification task. Normalization techniques (applied just before activation function) - Batch Norm, Instance Norm, Batch-Instance Norm, Layer Norm, and Group Norm are compared. All are hard-coded in pytorch ([scripts](https://github.com/sm354/COL870-Assignment-1/tree/main/ResNet%20and%20Normalizations)). In addition to hard-coded normalizations, batch norm of pytorch and without normalization (nn) variants of ResNet are also compared. 



![img](https://lh3.googleusercontent.com/ojhv6r8p3G8tnAMOtiII-4heIE2UL57OIfJLVYyw6Q5LVNmuugUrJmY1MoNCVmAJVRWMKkGe2dkUNdKuldXqJginRdrdPg0pHVRKd_dI8Y1ebYr_6_dmOC6wV1MK5q80IskOG6PN)

<img width="800" alt="Screenshot 2021-05-27 at 11 28 30 PM" src="https://user-images.githubusercontent.com/50492433/119874604-6ac5e400-bf43-11eb-925e-d5eca42cd2d6.png">

We also compare the **feature evolution** throughout learning:-

No Normalization            |  Batch-Instance Normalization normalization
:-------------------------:|:-------------------------:
![img](https://lh5.googleusercontent.com/VSiC9ONIpHZCW1E91nf92QbwdA4XV4PHDLNCWy4U74JEGjlg6CbQYA3GT-ZZiOzFS8aIuU91NjO288aGqG9Ca4VoU2ibsgjTqCvIL6K3z9TRHGHsRkssa752mp-hUuH7zwboN27u) |  ![img](https://lh6.googleusercontent.com/sIpydqM5F_1owAH7c_d_vC_W5BmlR0mzW8IBw93PC9bNwCELnojq6fJyy5XovA1fHvITqqVCQ1tVbjFiTCcJH5uhPF0wzD6V1Y8ekVUWxUVrleSi6zoM56yJcBreYRMz5Osm3JO0)



### Quick Start

To replicate the above results, follow the instructions in this section. Otherwise, you may skip to next section.

#### Training

```bash
cd ResNet\ and\ Normalizations
python train_cifar.py --normalization <norm_type> --output_file <path-to-save-model> --n 2 --num_epochs 100
```

- **normalization** : either one of the below
  - torch_bn : batch norm of pytorch. Remaining ones are implemented from scratch
  - bn : batch norm 
  - in : instance norm
  - bin : batch-instance norm
  - ln : layer norm
  - gn : group norm
  - nn : no normalization
- ***n*** : number of residual blocks (of same number of channels). Total layers of the model will be 6n+2.
- ***output_file*** is the path to save the trained model.

#### Testing

```bash
cd ResNet\ and\ Normalizations
python test_cifar.py --model_file <path-to-saved-model> --normalization <norm_type> --test_data_file <path-to-test-data.csv> --output_file <path-to-save-model-predictions.csv>
```



## Named Entity Recognition (NER) using Bi-LSTM

In the second part of the assignment, we use [[Lample et al., 2016](https://arxiv.org/abs/1603.01360)] for NER tagging on the publicly available GMB dataset. The objective here is to compare different variants of the model, and justify the performance gains as the complexity, or the prior we impose on the model, of the model increases. Following are the variants:

1. Bi-LSTM with randomly initialised word embeddings
2. Bi-LSTM with pretrained Glove word embeddings
3. Bi-LSTM with Glove word embeddings and character embeddings
4. Bi-LSTM with Glove word embeddings, character embeddings, and Layer Normalization (hard-coded LSTM for it)
5. Bi-LSTM with Glove word embeddings, character embeddings, Layer Normalization, and a CRF layer at the top.

We report the gain in performance when going from a simple Bi-LSTM (without pretrained word embeddings) to modifying our architecture and word embeddings to finally using a Bi-LSTM with CRF layer which gave best results. The scripts can be found [here](https://github.com/sm354/COL870-Assignment-1/tree/main/NER%20Tagging%20with%20BiLSTM).

<img width="800" alt="Screenshot 2021-05-27 at 11 34 55 PM" src="https://user-images.githubusercontent.com/50492433/119875924-f3914f80-bf44-11eb-8778-cd59fcf1fc0d.png">

### Quick Start

To replicate the above results, follow the instructions in this section. 

**[Download](https://nlp.stanford.edu/projects/glove/)**  Glove Embeddings named `glove.6B.100d.txt` 

#### Training

```bash
# go to NER Tagging with BiLSTM
cd NER\ Tagging\ with\ BiLSTM

# unzip data
tar -xf ner-gmb.tar.gz #will create a folder called "ner-gmb" having 3 files, test.txt, train.txt and dev.txt

# start training
python3 train_ner.py --initialization glove --char_embeddings 1 --layer_normalization 0 --crf 0 --output_file output_model.pth --data_dir ./NER_Dataset/ner-gmb --glove_embeddings_file ./glove_data/glove.6B.100d.txt --vocabulary_output_file output_vocab.vocab
```

- ***char_embeddings*** - set to 1 if you want to use character level embeddings
- ***layer_normalization*** - set to 1 if you want to use layer_normalized lstm cell in the BiLSTM
- ***crf*** - set to 1 if you want to use crf layer on top of the BiLSTM
- ***output_file*** output path for saving the trained model
- ***data_dir*** cleaned ner-data path, for example, the one which we get after unzipping ner-gmb.tar.gz
- ***glove_embeddings_file*** path to glove embeddings file "glove.6B.100d.txt"
- ***vocabulary_output_file*** output vocabulary file, created inside the training script only, which will be used while testing

#### Testing

```bash
python3 test_ner.py --model_file output_model.pth --char_embeddings 1 --layer_normalization 0 --crf 0 --test_data_file ./ner-gmb/test.txt --output_file output_predictions.txt --glove_embeddings_file ./glove_data/glove.6B.100d.txt --vocabulary_input_file output_vocab.vocab
```

- ***model_file*** is the same trained model which we generated using the train script above
- ***vocabulary_input_file*** is the same vocabulary file which we generated using the train script above

## References

[[1]](http://arxiv.org/abs/1607.06450): Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer Normalization.

[[2]](https://doi.org/10.1109/CVPR.2016.90): Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pages 770–778. IEEE Computer Society, 2016. doi: 10.1109/CVPR.2016.90.

[[3]](https://proceedings.neurips.cc/paper/2018/hash/018b59ce1fd616d874afad0f44ba338d-Abstract.html): Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. 32nd International Conference on Machine Learning, ICML 2015, 1:448–456, 2015. Hyeonseob Nam and Hyo-Eun Kim. Batch-instance normalization for adaptively style-invariant neural networks. In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicol`o Cesa-Bianchi, and Roman Garnett, editors, Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montr´eal, Canada, pages 2563–2572, 2018.

[[4]](http://arxiv.org/abs/1409.1556): Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015.

[[5]](http://arxiv.org/abs/1607.08022): Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Instance Normalization: The Missing Ingredient for Fast Stylization. (2016), 2016.

[[6]](): Yuxin Wu and Kaiming He. Group Normalization. International Journal of Computer Vision, 128(3):742–755, 2020.

















