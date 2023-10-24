# brainlm
Larger Language Models Converge on Brain-Like Representations of the World


# FMRI datasets

| Dataset Name                                                                     | Participants | Language        | Format                | Total n of words (cased / uncased) |
|----------------------------------------------------------------------------------|--------------|-----------------|-----------------------|------------------------------------|
| [Nouns](https://www.cs.cmu.edu/afs/cs/project/theo-73/www/science2008/data.html) | 9            | English         | seperate nouns        | 60                                 |
| [Harry Potter](http://www.cs.cmu.edu/~fmri/plosone/)                             | 8            | English         | book chapter          | 1405 / 1291                        |
| [Alice ](https://openneuro.org/datasets/ds002322/versions/1.0.3)                 | English      | book chapter    | 16                    | -                                  |
| [Pereira](https://osf.io/crwz7/)                                                 | 16           | English         | sentences             | -                                  |
| [Danders ](https://data.donders.ru.nl/collections/di/dccn/DSC_3011020.09_236?0)  | 204          | Dutch           | text                  | -                                  |
| [The Little Prince ](https://openneuro.org/datasets/ds003643/versions/2.0.1)     | 112          | English, French | book chapter          | -                                  |
| [Natural Stories](https://osf.io/eq2ba/?view_only=)                              | 19           | English         | natural story stimuli | 5228                               |
| [THINGS ](https://www.biorxiv.org/content/10.1101/2022.07.22.501123v1.abstract)  | -            | -               | -                     | -                                  |
| [BOLD](https://www.biorxiv.org/content/10.1101/2022.09.22.509104v1.full.pdf)     | -            | -               | -                     | -                                  |

## Experiments

**Attention**:

It is **not** necessary to process fmri data, build dictionaries, get LMs representations every time when running
pipeline.py. Please note that those data can be obtained once and stored for future use, which will save time and
resources.

run single bert model (In this case, don't use SBATCH --array=0-5)

```bash
python3 pipeline.py --config configs/bert_uncased_L-2_H-128_A-2.yaml
```

You can update any parameters in the config by adding --<parameter_name> <value>. For example:

```bash
python3 pipeline.py --config configs/bert_uncased_L-2_H-128_A-2.yaml --model:is_avg False
```

To run different LMs, you can update the config as following:

```bash
python3 pipeline.py --config configs/bert_uncased_L-2_H-128_A-2.yaml --model:model_alias gpt2 --model:model_name gpt2 --model:pretrained_model gpt2 --model:dim 768 model:n_layer 13
```

Every experiment config will be saved in expdir after finishing that run.

# RSA
To run RSA in natural stories dataset:
```bash
python3 pipeline.py --config nat_configs/fasttext.yaml --method rsa
```

To run RSA in Harry Potter dataset:
```bash
python3 pipeline.py --config hp_configs/fasttext.yaml --method rsa
```
