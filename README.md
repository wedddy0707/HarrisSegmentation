# On the Word Boundaries of Emergent Languages Based on Harris's Articulation Scheme
The implementation code of "[On the Word Boundaries of Emergent Languages Based on Harris's Articulation Scheme.](https://openreview.net/forum?id=b4t9_XASt6G)" ICLR 2023.

The word segmentation algorithm can be found at `boundary_detection.EntropyCalculator`.

This repo depends on EGG toolkit.
So, first of all, install EGG toolkit:
```
$ pip install git+ssh://git@github.com/facebookresearch/EGG.git
```
For more detail, refer to https://github.com/facebookresearch/EGG.

For training agents,
```
$ python train.py [options] > output_file.txt
```
To check options,
```
$ python train.py --help
```
For applying boundary detection,
```
$ python boundary_detection.py --config [configuration_file]
```
Format \& example of configuration file:
```yaml
log_dirs: [
    output_file_1.txt
    ...
    output_file_n.txt
]
least_acc: 0.9  # successful language
trained_lang_mode: "max_epoch"  # Option["max_epoch", "min_epoch_to_reach_least_acc"]
img_dir: "./img_dir"  # Directory to save png files
tmp_dir: "./tmp_dir"  # Directory to save temporary files
```
Configuration files have to be made in yaml format.
