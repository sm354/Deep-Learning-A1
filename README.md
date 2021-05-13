# COL870 Deep Learning : Assignment 1

In this assignment, we do experimental work to report

1. the impact of different normalization schemes like batch norm, instance norm, group norm, etc in deep CNN models (using CIFAR-10).
2. the gain in performance when going from a simple Bi-LSTM (without pretrained word embeddings) to a Bi-LSTM with CRF layer. 

The problem statement is given [here](https://github.com/sm354/COL870-Assignment-1/blob/main/dl2021ass1.pdf) and detailed report can be found [here](https://github.com/sm354/COL870-Assignment-1/blob/main/Report.pdf). 

## ResNet and various Normalization Schemes

We use ResNet model [[He et al., 2016](https://arxiv.org/abs/1512.03385)] to solve image classification task. Normalization techniques (applied just before activation function) - Batch Norm, Instance Norm, Batch-Instance Norm, Layer Norm, and Group Norm are compared. All are hard-coded in pytorch ([scripts](https://github.com/sm354/COL870-Assignment-1/tree/main/ResNet%20and%20Normalizations)). In addition to hard-coded normalizations, batch norm of pytorch and without normalization (nn) variants of ResNet are also compared. 



![img](https://lh3.googleusercontent.com/ojhv6r8p3G8tnAMOtiII-4heIE2UL57OIfJLVYyw6Q5LVNmuugUrJmY1MoNCVmAJVRWMKkGe2dkUNdKuldXqJginRdrdPg0pHVRKd_dI8Y1ebYr_6_dmOC6wV1MK5q80IskOG6PN)

![img](https://lh3.googleusercontent.com/QpXYRudfXJX4cSszgMIgJZv9pxGIiEP1dYW6K9d6Lc9gHHHeozYlA-Q570jV3yXlhXFlC28xQfuP-gu1zwVMKJCX9oY7KtOAfjV-E8-7wgc7evpqPp8Az7XNxfYW8Ho13Uf_rH1B)

We also compare the feature evolution throughout learning:-



![img](https://lh5.googleusercontent.com/VSiC9ONIpHZCW1E91nf92QbwdA4XV4PHDLNCWy4U74JEGjlg6CbQYA3GT-ZZiOzFS8aIuU91NjO288aGqG9Ca4VoU2ibsgjTqCvIL6K3z9TRHGHsRkssa752mp-hUuH7zwboN27u)

![img](https://lh6.googleusercontent.com/sIpydqM5F_1owAH7c_d_vC_W5BmlR0mzW8IBw93PC9bNwCELnojq6fJyy5XovA1fHvITqqVCQ1tVbjFiTCcJH5uhPF0wzD6V1Y8ekVUWxUVrleSi6zoM56yJcBreYRMz5Osm3JO0)

## Named Entity Recognition (NER) using Bi-LSTM

In the second part of the assignment, we use [[Lample et al., 2016](https://arxiv.org/abs/1603.01360)] for NER tagging on the publicly available GMB dataset. The objective here is to compare different variants of the model, and justify the performance gains as the complexity, or the prior we impose on the model, of the model increases. Following are the variants:

1. Bi-LSTM with randomly initialised word embeddings
2. Bi-LSTM with pretrained Glove word embeddings
3. Bi-LSTM with Glove word embeddings and character embeddings
4. Bi-LSTM with Glove word embeddings, character embeddings, and Layer Normalization (hard-coded LSTM for it)
5. Bi-LSTM with Glove word embeddings, character embeddings, Layer Normalization, and a CRF layer at the top.

![img](https://lh3.googleusercontent.com/fS3zoIEsb8Xtp4w7-Bha0yLJnIjns5HsmFp5h2kM1xTYMEmmUrYUCcRp-TTbGXak93L0WRBXaT4nXX_Uio5cG7b-2iBgXcYYGco1AoPjArjUQoRrGoViCTHumwBnTb8np4oUf77y)

The scripts can be found [here](https://github.com/sm354/COL870-Assignment-1/tree/main/NER%20Tagging%20with%20BiLSTM).















