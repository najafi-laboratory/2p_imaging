# MC11 Scratch Resume Run - 2026-05-11

Manifest:

- `slurm_runs/mc11_scratch_remaining_20260511.csv`

Output root:

- `/storage/scratch1/3/grubin6/2p_processing_results`

The project-storage MC11 outputs under `/storage/project/r-fnajafi3-0/shared/2P_Imaging/MC11_Processed` are read-only to this account, so the active resumed run targets the writable scratch results tree.

## Stage Markers

| Stage | Output markers tracked |
| --- | --- |
| `prep` | `raw_voltages.h5`, `suite2p/plane0/ops.npy` |
| `suite2p` | `suite2p/plane0/F.npy`, `Fneu.npy`, `spks.npy`, `stat.npy`, `iscell.npy`, `redcell.npy` |
| `qc` | `qc_results/fluo.npy`, `neuropil.npy`, `stat.npy`, `masks.npy`, `move_offset.h5`, `ops.npy` |
| `label` | `masks.h5` |
| `dff` | `dff.h5` |

## Submitted Jobs

| Session | Submitted stages |
| --- | --- |
| `MC11_20260317_2afc_PFC-1326` | `dff=8367740` |
| `MC11_20260318_2afc_PFC-1329` | `dff=8367741` |
| `MC11_20260319_2afc_PFC-1332` | `dff=8367742` |
| `MC11_20260320_2afc_PFC-1335` | `dff=8367743` |
| `MC11_20260324_2afc_PFC-1342` | `prep=8367744`, `suite2p=8367745`, `qc=8367746`, `label=8367747`, `dff=8367748` |
| `MC11_20260327_2afc_PFC-1359` | `prep=8367749`, `suite2p=8367750`, `qc=8367751`, `label=8367752`, `dff=8367753` |
| `MC11_20260330_2afc_PFC-1377` | `prep=8367754`, `suite2p=8367755`, `qc=8367756`, `label=8367757`, `dff=8367758` |

Check status with:

```bash
python -m utils_2p.slurm_pipeline status --children /storage/scratch1/3/grubin6/2p_processing_results
```
