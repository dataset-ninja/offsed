**OFFSED: Off-Road Semantic Segmentation Dataset** is a dataset for instance segmentation, semantic segmentation, object detection, and identification tasks. It is used in the automotive industry. 

The dataset consists of 1020 images with 8586 labeled objects belonging to 20 different classes including *obstacle*, *person*, *tree*, and other: *grass*, *drivable dirt*, *sky*, *bush*, *nondrivable dirt*, *building*, *held object*, *wall*, *drivable pavement*, *crops*, *excavator*, *nondrivable pavement*, *car*, *guard rail*, *truck*, *camper*, and *background*.

Images in the OFFSED dataset have pixel-level instance segmentation annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation (only one mask for every class) or object detection (bounding boxes for every object) tasks. There are 817 (80% of the total) unlabeled images (i.e. without annotations). There are no pre-defined <i>train/val/test</i> splits in the dataset. The dataset was released in 2021 by the University of Kaiserslautern, Germany.

Here are the visualized examples for the classes:

[Dataset classes](https://github.com/dataset-ninja/offsed/raw/main/visualizations/classes_preview.webm)
