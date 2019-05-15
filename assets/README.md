# Dataset Format

The dataset folder is organized as follows:

```
center_single/
    RAVEN_0_train.npz
    RAVEN_0_train.xml
    ...
    RAVEN_6_val.npz
    RAVEN_6_val.xml
    ...
    RAVEN_8_test.npz
    RAVEN_8_test.xml
    ...
distribute_four/
    ...
distribute_nine/
    ...
in_center_single_out_center_single/
    ...
in_distribute_four_out_center_single/
    ...
left_center_single_right_center_single/
    ...
up_center_single_down_center_single/
    ...
```

Note that each npz file comes with an xml file.

These 7 folders correspond to the 7 figure configurations in the paper. Specifically,

* Center = center_single
* 2x2Grid = distribute_four
* 3x3Grid = distribute_nine
* Left-Right = left_center_single_right_center_single
* Up-Down = up_center_single_down_center_single
* Out-InCenter = in_center_single_out_center_single
* Out-InGrid = in_distribute_four_out_center_single

## Naming

You might notice that the actual naming in this dataset is slightly different from what's reported in our paper. This is mostly due to the fact that things like **2x2** or **3x3** do not have corresponding word vectors. They are now **distribute_four** and **distribute_nine**. To make the paper concise, we also remove certain adjectives. **Center** was **Center_Single** and sometimes came with a component name. 

As described in the paper, embeddings for each of them are obtained from pre-trained GloVe vectors and held fixed during training.

## NPZ file

Each npz file contains the following:

* image: a (16, 160, 160) array where all 16 figures in each problem are stacked on the first dimension. Note that first 8 figures compose the problem matrix and the last 8 figures are choices.
* target: the index of the correct answer in the answer set. Note that it starts from 0 and you should offset it by 8 if you want to retrieve it from the image array.
* structure: the tree structure annotation for the problem. It's serialized into a sequence using pre-order traversal.
* meta_matrix: similar to that in PGM. Detailed ordering could be found in ```src/dataset/const.py```.
* meta_target: bitwise-or of meta_matrix on all rows. 
* meta_structure: it's similar to meta_matrix. Detailed ordering is in ```src/dataset/const.py```.

## XML file

Each xml file contains the following:

* Context panels and choice panels: each Panel could be further decomposed into Struct, Component, Layout, and Entity.
  * Each layer comes with its name and id if necessary.
  * Layout has its own attributes, whose values are indices into the value set (see also ```src/dataset/const.py```), except Position. Position is a list of slots entities could occupy, denoted by center and width/height. 
  * Entity's attributes follow the same annotation. The bbox is retrieved from the Position array in its parent Layout and the real_bbox is the actual bounding box, denoted by center and width/height. The mask is encoded using the run-length encoding. To decode it, use the ```rle_decode``` function in ```src/dataset/api.py```.
* Rules: rules are divided into groups, each of which applies to the corresponding component with the same id number. 
  * ```attr``` could be ```Number/Position``` when the rule is ```Constant``` as these two attributes are deeply coupled.
  * When there is a rule on ```Number``` or ```Position```, we omit the rule on the other attribute, as it should be assumed **as is**, *i.e.*, following the rule on the other (could remain unchanged).
  * Therefore, each rule group has 4 rules.