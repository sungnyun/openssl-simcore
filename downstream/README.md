## Downstream tasks

We provide the training and evaluation codes for the downstream tasks which we have experimented on.

### Summary

These tasks are outlined as follows.    
**Task name: `path`, `reference in the main paper`**

- object detection: `downstream/detection/`, `Table 7c`
- pixel-wise segmentation: `downstream/segmentation/`, `Table 7c`
- open-set semi-supervised learning: `downstream/opensemi/`, `Table 6`
- webly supervised learning: `downstream/weblysup/`, `Table 6`
- semi-supervised learning: `downstream/semisup/`, `Table 15 in Appx. F.3`
- active learning: `downstream/active/`, `Table 16 in Appx. F.3`
- hard negative mining: `downstream/mining/`, `Table 17 in Appx. G`


Note: _Semi-supervised learning here includes MixMatch, ReMixMatch, FixMatch, and FlexMatch, all of which utilize both labeled and unlabeled target data.
On the other hand, semi-supervised learning in [train_sup.py](../train_sup.py) utilizes only partially labeled data (Table 7b in the paper), 
which follows the same protocol as other SSL works._

### Run
Every running files are located in each task. For example, to run the OpenSemi experiment with OpenMatch method, run
```sh
$ cd downstream/opensemi/
$ bash run_openmatch.sh
```
All you have to do is setup the **dataset** and its **path**, and the **pretrained model checkpoint**.
