# MonkeyProject
> Classification for an image dataset of 10 Monkey Species

## Introduction

This project is an image classification task with a dataset consisting of monkey images of 10 species. The aim of this contest is to create a classifier to classify monkeys by 10 species. After training the training set data, the classifier model could be used to classify the types of monkeys according to the images of the monkeys in the test set. Finally, the accuracy of the classification prediction results on the test set will be used as the criterion to judge the classifier.

The selected contest comes from：https://www.kaggle.com/slothkong/10-monkey-species

## Dataset

The data set comes from image data provided by Kaggle's competition project. The dataset consists of two file, the training dataset and the test dataset. There are 10 classes of images in this dataset. The dataset consists of almost 1400 images, which are 400 x 300 px or lager in JPEG format.

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

The size of the image with the shortest edge: 183