# Structural Similarities Between Language Models and Neural Response Measurements

This is the code to replicate the experiments described in the paper (to appear in NeurReps@NeurIPS, 2023):

> Jiaang Li*, Antonia Karamolegkou*, Yova Kementchedjhieva, Mostafa Abdou, Sune Lehmann, and Anders Søgaard. [Structural Similarities Between Language Models and Neural Response Measurements.](https://openreview.net/forum?id=ZobkKCTaiY) In _NeurIPS 2023 Workshop on Symmetry and Geometry in Neural Representations 2023_.

## Installation
You can clone this repository issuing: <br>
`git clone git@github.com:coastalcph/brainlm.git`

1\. Create a fresh conda environment and install all dependencies.
```text
conda create -n brain2lang python=3.11
conda activate brain2lang
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```

## fMRI datasets

| Dataset Name                                                                     | Participants | Language        | Format                | Total n of words |
|----------------------------------------------------------------------------------|--------------|-----------------|-----------------------|------------------------------------|
| [Harry Potter](http://www.cs.cmu.edu/~fmri/plosone/)                             | 8            | English         | book chapter          | 1405                        |
| [Natural Stories](https://osf.io/eq2ba/?view_only=)                              | 19           | English         | Natural story stimuli | 5228                               |

## How to run

See available model configurations in [`config.py`](./src/config.py) under `MODEL_CONFIGS` and available saving paths of datasets, runtime parameters, and projection method in [`config.py`](./src/config.py) under `RunConfig`.

Example to sequentially run BERT-Tiny and BERT-Mini models utilizing the Procrustes Analysis method on the Harry Potter dataset:

```bash
python main.py \
    --multirun \
    models=bert-tiny,bert-mini \
    datasets=hp_fmri \
    projection_method=Procrustes
```

## How to Cite

```bibtex
@inproceedings{
li2023structural,
title={Structural Similarities Between Language Models and Neural Response Measurements},
author={Jiaang Li and Antonia Karamolegkou and Yova Kementchedjhieva and Mostafa Abdou and Sune Lehmann and Anders S{\o}gaard},
booktitle={NeurIPS 2023 Workshop on Symmetry and Geometry in Neural Representations},
year={2023},
url={https://openreview.net/forum?id=ZobkKCTaiY}
}
```