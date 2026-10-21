# Attribution

- **DeepLSD model and pretrained weights:** Rémi Pautrat (2022), MIT. Source: https://github.com/cvg/DeepLSD, commit `f7d9d6258c0cd25d4f6eea882853565403d289be`. License: `licenses/DeepLSD-MIT.txt`. Checkpoint source and hashes: `weights/torch/provenance.json`.
- **C++ pipeline:** `cpp/include/` and `cpp/src/` use AGPL-3.0-or-later; full license: `licenses/AGPL-3.0.txt`. The legacy local extractor and Python binding have been removed. Upstream pytlsd/LSD source and bindings are not included. The LSD citation below acknowledges the algorithmic background.
- **TUM RGB-D images:** CC BY 4.0 unless stated otherwise by the authors. Source and license: https://cvg.cit.tum.de/data/datasets/rgbd-dataset . The showcase videos and `assets/demo.gif` add detected-line and performance overlays. Their footer credits the dataset and CC BY 4.0 license. `data/sample.mp4` is an unannotated 30-second RGB excerpt; simple overlay outputs carry this attribution through the README and this notice. Source frames and all derived footage retain this dataset attribution.
- Installed Python packages and Axelera SDK/runtime retain their own licenses. Proprietary SDK libraries are not included here.

DeepLSD citation: R. Pautrat, D. Barath, V. Larsson, M. R. Oswald and M. Pollefeys, “DeepLSD: Line Segment Detection and Refinement with Deep Image Gradients,” CVPR 2023.

LSD citation: R. Grompone von Gioi, J. Jakubowicz, J.-M. Morel and G. Randall, “LSD: a Line Segment Detector,” IPOL 2012, DOI:10.5201/ipol.2012.gjmr-lsd.

TUM citation: J. Sturm, N. Engelhard, F. Endres, W. Burgard and D. Cremers, “A Benchmark for the Evaluation of RGB-D SLAM Systems,” IROS 2012.
