# Third-party software

The inference subset in `src/flexgen/_vendor` retains upstream copyright headers and license texts:

- BBFlow-derived backbone components (`src/flexgen/_vendor/frame_flow`): Max Planck Institute for Polymer Research, HITS and Microsoft; MIT (`BBFLOW_LICENSE`).
- GAFL: HITS and Microsoft; MIT, with BSD terms for GATr-derived components (`GAFL_LICENSE`).
- GATr: Qualcomm Technologies; BSD 3-Clause (`GATR_LICENSE`).
- OpenFold and AlphaFold-derived geometry: AlQuraishi Laboratory and Google DeepMind; Apache 2.0 (`APACHE_2_0_LICENSE`). This project includes software from https://github.com/aqlaboratory/openfold.

Vendor modifications isolate inference import namespaces and resource loading, expose final backbone features, accept latent-derived node context, and retain only the conditional inference prior and required utilities. Original notices remain applicable to modified files. Stereo-chemical tables and geometric-algebra product tensors accompany their source modules.

Proteina-Complexa source and weights are external dependencies, distributed by NVIDIA under their respective terms. They are not bundled or relicensed here. These third-party notices do not grant a license to the project's original code or model weights.
