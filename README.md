# ShT5-HierGen: A Shallow T5 Model Designed for Hierarchical Generation

This repository provides the code for **ShT5-HierGen**, a variant of the vanilla T5 model specifically designed for improved performance on Multi-label and Single-label Hierarchical Text Classification tasks. Key features include:

- **Single-layer decoder** for streamlined architecture.
- **Reduced output vocabulary** by replacing the original Byte-Pair Encoding (BPE) tokenizer with a character-based tokenizer.

The repository also includes an implementation of the vanilla T5 model, following the work by Torba et al. (2024), and all code necessary to reproduce the experiments described in the associated paper.

**Supported Datasets:**
- [**Web of Science (WOS)**](https://data.mendeley.com/datasets/9rw3vkcfy4/6) 
- [**Blurb Genre Collection (BGC)**](https://aclanthology.org/P19-2045/)
- [**Medical Device Coding Failure (MDCF):**](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfmaude/search.cfm)
  Extracted from the U.S. Food and Drug Administration Manufacturer and User Facility Device Experience (MAUDE) database, this dataset contains reports describing failures related to medical devices.

If you build upon this work in your research, please cite the following paper:

```bibtex
@article{gibelloDataDriven2024,
  title={ShT5-HierGen: Leveraging Shallow Decoding for Efficient Hierarchical Sequence Generation},
  author={Gibello, Riccardo and Ren, Yijun and Caiani, Enrico Gianluca},
  journal={},
  year={2025}
}
```

We welcome contributions and feedback!

- Feel free to fork this repository and submit pull requests to contribute to the project.
- For detailed guidelines, please see the [CONTRIBUTING section](./CONTRIBUTING.md).
- If you have any questions or encounter issues, contact us at [riccardo.gibello@polimi.it](mailto:riccardo.gibello@polimi.it) or open a GitHub issue.

## Requirements

- **Python 3.13** is required. You can download it from the [official Python website](https://www.python.org/downloads/).
- All other Python dependencies will be installed automatically by running the `setup_environment.py` script (see the [Installation](#installation) section).

> **Note:**  
> There are no strict hardware requirements to run the experiments. However, we recommend using a machine with at least 8GB of VRAM (such as the GeForce 3070 used by the authors) for optimal performance.

## Installation

To set up the Python environment, open a terminal and run:

```bash
python .\setup_environment.py
```

This script will create a virtual environment in the project root and install all required packages.
The process may take a few minutes. GPU-accelerated packages will be installed automatically if a compatible GPU is detected.

## Input directory setup & Pipeline Execution

During the execution, the pipeline will track the amount of carbon emissions produced through the Python package
`codecarbon`. Every output file for each experiment will be saved in the `data/_emissions/` folder. Please, refer to the [CodeCarbon documentation](https://mlco2.github.io/codecarbon/) for details on how to interpret the results.

## License

This project’s source code is licensed under the GNU General Public License (GPL), version 3 or later. See the full
license text in the [GNU_LICENSE.txt](./GNU_LICENSE.txt) file. For more information, please refer to the
[LICENSE](./LICENSE.md) page.

## Code of Conduct

This project has adopted
the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct.html). By
participating in this project, you agree to abide by its terms. For more information, please refer to the
[CODE_OF_CONDUCT](./CODE_OF_CONDUCT.md) page.
