# Leakage-Aware IMU-Based Human Posture Recognition

Code and results accompanying the paper *"Leakage-Aware IMU-Based Human Posture Recognition with Block-Wise Evaluation and a Simulated Flex-Sensor Probe"* (ARIIA-2026).

## What this repo is (and isn't)

- **Part A (real contribution):** a leakage-aware benchmark for IMU-based posture recognition on real sensor data — quantifying how much naive train/test splitting inflates accuracy, and disclosing a silent class-undersampling failure mode caused by non-stratified block splitting.
- **Part B (disclosed feasibility probe, not a real result):** a synthetic flex-sensor feature used only to test whether the feature-engineering/explainability pipeline (SHAP, LIME) is fusion-ready. No accuracy improvement claim is made — the feature is derived from the target label, so any accuracy gain is circular by construction.
- **Part C (disclosed negative result):** a change-point-gated fusion approach for next-posture forecasting, evaluated honestly against a persistence baseline on the dataset's five real posture transitions. The result is negative and is reported as such.

## Dataset

We use the publicly available `imu.dat` dataset:

> H. Kale, P. Mandke, H. Mahajan, and V. Deshpande, "Human Posture Recognition using Artificial Neural Networks," in *2018 IEEE 8th International Advance Computing Conference (IACC)*, Greater Noida, India, 2018, pp. 272–278. https://doi.org/10.1109/IADCC.2018.8692143

Dataset source: [pkmandke/Human-Posture-Dataset](https://github.com/pkmandke/Human-Posture-Dataset)

- 44,800 samples, 6 posture classes (Sleeping, Standing, Sitting, Running, Forward Bending, Backward Bending)
- Two MPU-6050 IMUs (chest + thigh), pooled recordings from 3 subjects (indistinguishable in the released data)
- ~5 Hz effective sampling rate (200 ms transmission interval)
- Licensed GPL v3 by the original authors — please cite their paper if you use the dataset

## Repo structure

```
├── notebooks/
│   └── posture_recognition_leakage_aware_PARTC_v2.ipynb   # full pipeline, Parts A–C
├── results/
│   ├── leakage_ablation_naive_vs_block_split.csv           # Part A headline leakage numbers
│   ├── ablation_results.csv                                 # RF/GB/Ensemble, with/without flex
│   ├── lstm_forecast_ablation.csv                            # Part C forecasting results
│   ├── shap_flex_attribution_by_class.csv
│   ├── lime_explanations.json
│   └── *.png                                                 # confusion matrices, SHAP plots, training curves
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Run the notebook in **Google Colab** (recommended) or a local Jupyter environment with unrestricted internet access — it needs to download `imu.dat` via `gdown` and `pip install` TensorFlow/SHAP/LIME.

## Reproducing the headline results

The notebook is organized to be run top-to-bottom:

1. **Sections 1–6** — download, clean, and feature-engineer the dataset (chronological order preserved, no shuffling — this matters for the leakage analysis).
2. **Section 7 / 7a** — the core leakage-aware split (`StratifiedGroupKFold` over 500-row blocks) and the naive-vs-block-split accuracy comparison (Part A headline result).
3. **Sections 8–13** — threshold baseline, RF/GB/Ensemble with and without the synthetic flex feature.
4. **Sections 14–15** — SHAP/LIME explainability on the flex-fusion-ready pipeline (Part B).
5. **Sections 16+** — change-point-gated fusion for next-posture forecasting (Part C).

All random seeds are fixed (`RANDOM_STATE`, defined in Section 1) for reproducibility of the reported numbers.

## Key results

| Split strategy | Accuracy |
|---|---|
| Naive row-level split (leaky) | 99.85% |
| Block-stratified split (honest) | 99.56% |

Gap attributable to temporal leakage: **0.30 percentage points**. A naive *random* (non-stratified) block split additionally risks dropping the minority class (Running) entirely from the test set — see the paper's Results section for details.

## Limitations

- Single publicly available dataset; recordings from 3 subjects are pooled and indistinguishable in the released data, so subject-level (e.g. LOSO) validation isn't possible.
- The flex-sensor feature is synthetic, not measured — a feasibility probe only.
- Only 5 real posture transitions exist in the dataset, limiting the transition-forecasting evaluation (Part C) to a small sample.

## Citation

If you use this code, please cite our paper (details to be added on acceptance) and the original dataset paper by Kale et al. above.

## License

Code in this repository is released under [choose: MIT / GPL v3 — match the dataset's license if redistributing any dataset-derived files]. See `LICENSE` for details.
