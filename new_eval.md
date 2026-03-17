# New Evaluation Results
Seed=42, N=500 per split, img mode, evaluate.py (volumetric IoU + Chamfer Distance).

## Paper Targets (Table 2)
| Model | DeepCAD IoU | Fusion360 IoU |
|-------|-------------|---------------|
| cadrille SFT (paper) | 86.1% | 77.6% |
| cadrille Dr. CPPO (paper) | **92.2%** | **84.6%** |

## Our Results
| Model | DeepCAD IoU | DeepCAD CD | DeepCAD IR | Fusion360 IoU | Fusion360 CD | Fusion360 IR |
|-------|-------------|------------|------------|---------------|--------------|---------------|
| Official SFT| **86.9%** | 0.183 | 2.4%| **77.4%** | 0.205 | 2.0% |
| Official RL (paper)| **91.9%** | 0.162 | 0.0%| **84.1%** | 0.177 | 0.0% |
| Ours a100-step4500| **87.9%** | 0.182 | 0.4%| **79.9%** | 0.198 | 0.8% |
| Ours a100-step6000| **88.0%** | 0.177 | 0.6%| **79.9%** | 0.194 | 1.2% |
| Ours a100-step7200| **88.0%** | 0.174 | 1.0%| **80.0%** | 0.192 | 0.6% |

## Notes
- IR = Invalid Rate (fraction of samples where CadQuery execution failed)
- CD = median Chamfer Distance × 1000
- All evals: img mode, seed=42, 500 random samples from official test sets
- cad_ckpt = our RL checkpoints from A100 training run
