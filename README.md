# MonkeyProject
> An image classification challenge of 10 monkey sapecies

## Introduction

This project is an image classification challenge from Kaggle, [10-monkey-species](https://www.kaggle.com/slothkong/10-monkey-species), with a dataset consisting of monkey images of 10 species. The aim of this contest is to create a classifier to classify monkeys by 10 species. After training the training set data, the classifier model can be used to classify the class of per monkeys according to the images of the monkeys in the validation set. Finally, the accuracy of the classification prediction results on the validation set will be used as the criterion to judge the classifier.

## Dataset

The dataset is provided by [10-monkey-species](https://www.kaggle.com/slothkong/10-monkey-species). The dataset consists of the training set and the validation set. There are 10 classes of images in this dataset. The dataset consists of almost 1400 images, which are colored ones of 400 x 300 px or lager in jpg format.

The 10 species of monkeys are as follows：

| Label | Latin Name            | Common Name               | Train Images | Validation Images |
| ----- | --------------------- | ------------------------- | ------------ | ----------------- |
| n0    | alouatta_palliata     | mantled_howler            | 131          | 26                |
| n1    | erythrocebus_patas    | patas_monkey              | 139          | 28                |
| n2    | cacajao_calvus        | bald_uakari               | 137          | 27                |
| n3    | macaca_fuscata        | japanese_macaque          | 152          | 30                |
| n4    | cebuella_pygmea       | pygmy_marmoset            | 131          | 26                |
| n5    | cebus_capucinus       | white_headed_capuchin     | 141          | 28                |
| n6    | mico_argentatus       | silvery_marmoset          | 132          | 26                |
| n7    | saimiri_sciureus      | common_squirrel_monkey    | 142          | 28                |
| n8    | aotus_nigriceps       | black_headed_night_monkey | 133          | 27                |
| n9    | trachypithecus_johnii | nilgiri_langur            | 132          | 26                |
|       |                       |                           | 1098         | 272               |

The per-channel mean and per-channel standard deviation of the training data:

* mean: [0.4334, 0.4296, 0.3319]
* std: [0.2636, 0.2597, 0.2610]

abnormal data:

* `File is not .jpg ./10-monkey-species/training/n9/n9151jpg`

  rename this file to `n9164.jpg`

* `File is not .jpg ./../10-monkey-species/training/n9/n9160.png`

  rename this file to `n9160.jpg`
