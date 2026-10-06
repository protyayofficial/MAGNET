# MAGNET

**Manifold-Aware Graph Diffusion Network for Functional Brain Connectome Generation** (**NeurIPS 2026**)

MAGNET generates valid class-conditional correlation connectomes with a normalized Cholesky representation and a graph transformer denoiser. This public release contains the MAGNET implementation, scripts for the five Table 1 dataset/atlas settings, and MAGNET-only zero-shot evaluation. Please visit our [project page](https://protyayofficial.github.io/MAGNET/) for further details.

## Abstract

> Functional brain connectome represents neural connectivity as a matrix of pairwise interactions between brain regions. Generation of functional connectomes is not only a question of validity; rather, having satisfied the constraints on the correlation matrix, the next step is to recover the class-conditional geometry buried under coarse labels. We propose MAGNET, a Manifold-Aware Graph Diffusion Network which uses a normalized-Cholesky representation of the manifold of correlation matrices that guarantees validity. MAGNET lifts noisy latent states into ROI-level region tokens and performs denoising with a relational inductive bias over brain atlas regions. To deal with structural problems induced by coarse labels, MAGNET employs class-anchored conditioning, amortized structural bridge, and relevance-preserving corruption. Across ABIDE, ADNI, and OASIS-3, MAGNET consistently achieves favorable results compared to previous manifold-aware approaches, demonstrating improvements of $7-21$% in class-conditional fidelity ($\alpha,\beta$-F1) across three cohorts and better sampling efficiency. Moreover, while training only with strict binary labels, MAGNET is capable of maintaining clinical heterogeneity through fine substructure of the connectomes in a zero-shot setting, improving subclass covariance alignment ($\lambda$-MSE) by over 30%. These results suggest that geometric validity is a necessary but insufficient condition for clinical utility in connectome synthesis. Moreover, efforts in making the diffusion denoising class-conditional manifold aware finds utility beyond the highly curved brain connectome generation as this is a critical problem in various general settings.

## Scope

| Dataset | Atlas | Regions | MAGNET run |
|---|---|---:|---|
| ABIDE | MSDL | 39 | `results/table1/msdl/gdt` |
| ADNI | MSDL | 39 | `results/table1/msdl/gdt` |
| OASIS-3 | MSDL | 39 | `results/table1/msdl/gdt` |
| ABIDE | AAL | 116 | `results/table1/aal116/gdt` |
| ABIDE | CC200 | 200 | `results/table1/cc200/gdt` |

The generator is trained for 200 epochs per split with batch size 64. The main evaluation uses six sampling steps and 10 subject-disjoint splits with 10% held out. The same OAS covariance estimator and correlation normalization are used across methods. The AAL116 and CC200 ABIDE cohorts are quality-controlled separately, so their exact sample counts can differ from MSDL and from each other.

## Setup

```bash
conda create -n MAGNET python=3.11 pip -y
conda activate MAGNET
pip install -r requirements.txt
```

The `MAGNET` environment and the shipped metric implementation are sufficient for MAGNET training and MAGNET-only evaluation; no baseline checkout is required for these commands. Use a PyTorch build compatible with your CUDA driver. GPU use is optional, but the full five-setting run is computationally expensive.

The pinned Python dependencies match the environment used to smoke-test this release; the original baseline repositories may need additional packages. Record CUDA driver and GPU model when reporting timing results.

## Data

The MSDL loader automatically downloads the same public dataset archive used by DiffeoCFM on first use. It expects `data_atlas_msdl/{abide,adni,oasis3}_X_y.pkl`, with columns `SubjectID`, `Diagnosis`, and `TimeSeries`. No subject-level data are bundled here.

For the two dense ABIDE atlases, run:

```bash
python scripts/download_dense_abide.py
```

This uses Nilearn's ABIDE Preprocessed Connectomes Project fetcher: C-PAC pipeline, band-pass filtering, quality-checked subjects, and no global signal regression. It retains at least 30 frames, rejects ROI variance at or below `1e-12`, and intersects each atlas with the MSDL ABIDE reference cohort. It writes `data_atlas_aal116/abide_X_y.pkl` and `data_atlas_cc200/abide_X_y.pkl`. The paper cohorts contained 792 AAL116 and 831 CC200 subjects; check the printed counts before claiming an exact numerical reproduction, since the data host and Nilearn behavior can change.

## Run MAGNET

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_table1_magnet.sh
```

The script writes generated and real correlation matrices, labels, groups, split histories, training time, and sampling time under `results/table1/`. Use `PYTHON_BIN=/path/to/python` to select an interpreter. To reproduce a single setting, invoke `python src/train_gdt.py --help` and pass the paper settings shown in `scripts/run_table1_magnet.sh`.

## Table 1 baselines and evaluation

Baseline source code is deliberately **not** redistributed. We used the [official DiffeoCFM code](https://github.com/antoinecollas/DiffeoCFM) for DiffeoCFM, DiffeoGauss, and projected TriangCFM, and the [official GDSS code](https://github.com/harryjo97/GDSS) for the projected GDSS variant. We are especially grateful to the DiffeoCFM authors for releasing their data loading and evaluation code. Refer to those projects for their own licenses and baseline training commands. Our GDSS-proj experiment adapted its adjacency score network to signed, continuous connectomes and projected samples to valid correlation matrices; the upstream GDSS graph benchmark alone does not reproduce that variant.

For strict Table 1 comparison, use the same downloaded inputs, split indices, class labels, OAS correlation estimation, 200 epochs, and six sampling steps for every method. The MAGNET outputs follow DiffeoCFM's `split_*` array layout. Evaluate them with the included seeded alpha/beta metric implementation, which follows the EvaGeM protocol used by DiffeoCFM:

```bash
python scripts/evaluate_table1.py
```

If you have compatible split arrays from external baseline runs, add them to the comparison:

```bash
python scripts/evaluate_table1.py --baseline-results-dir /path/to/baseline/results
```

This writes `results/table1_metrics/split_metrics.csv` and `comparison.csv`. The adapter checks that real train/validation arrays and labels match across methods for each split, refusing a misleading comparison otherwise. Cloning unmodified DiffeoCFM alone does **not** reproduce the complete Table 1: its published configuration fixes the atlas to MSDL and its split RNG differs from MAGNET's train-only split seeds; AAL116/CC200 and exact split alignment need adaptation. The adapter accepts its raw triangular output and applies the published SPD projection when necessary. GDSS-proj additionally requires the signed-connectome adaptation not distributed here. To reproduce those rows, run suitably adapted external baselines and pass their matching split arrays to the adapter. Training/sampling times will vary with hardware.

## Zero-shot MAGNET

After generating MSDL ADNI and OASIS-3 outputs:

```bash
bash scripts/run_zero_shot_magnet.sh
```

The zero-shot evaluator keeps the original binary-trained MAGNET generator fixed. It uses detailed ADNI and OASIS-3 labels only for post hoc evaluation and writes split-level, class-level, and aggregate CSV files to `results/zero_shot_magnet/`. It uses the included seeded alpha/beta metrics; no baseline checkout or DiffeoCFM/GDSS results are required by this command.


## Credits

We sincerely thank authors of [DiffeoCFM](https://github.com/antoinecollas/DiffeoCFM) and [GDSS](https://github.com/harryjo97/GDSS) for the code release and well maintained code repositories. 
