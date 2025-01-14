
AbMEGD
![overview](./assets/overview.jpg)
Antibody Design and Optimization with Multi-scale Equivariant Graph Diffusion Models for Accurate Complex Antigen Binding

## Install

### Environment

```bash
conda env create -f env.yaml -n AbMEGD
conda activate AbMEGD
```

The default `cudatoolkit` version is 11.3. You may change it in [`env.yaml`](./env.yaml).

### Datasets and Trained Weight

Protein structures in the `SAbDab` dataset can be downloaded [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/). Extract `all_structures.zip` into the `data` folder.

The `data` folder contains a snapshot of the dataset index (`sabdab_summary_all.tsv`). You may replace the index with the latest version [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/).

## Design and Optimize Antibodies

3 design modes are available. Each mode corresponds to a config file in the `configs/test` folder:
| Config File | Description |
| ------------------------ | ------------------------------------------------------------ |
| `codesign_single.yml` | Sample both the **sequence** and **structure** of **one** CDR. |
| `codesign_multicdrs.yml` | Sample both the **sequence** and **structure** of **all** the CDRs simultaneously. |
| `abopt_singlecdr.yml` | Optimize the **sequence** and **structure** of **one** CDR. |

## Train

```bash
python train.py ./configs/train/<config-file-name>
```
