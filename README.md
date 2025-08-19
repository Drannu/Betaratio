# Betaratio – Code for *Investigating Circumstellar Atomic Radiation-driven Dynamics* (Lehtmets et al. 2025)

This repository contains the code and supporting material for the article  
**“Investigating Circumstellar Atomic Radiation-driven Dynamics”**  
by *Lehtmets, A., Kama, M., Fossati, L., & Aret, A.* (2025).

The paper studies how **stellar radiation pressure and gravity influence atomic species** in circumstellar environments. We calculate the radiation-to-gravitational force ratio (𝛽), evaluate the velocity boost of neutral atoms before ionisation, and explore implications for **stellar photospheric contamination** by gas from **debris discs, evaporating hot Jupiters, and rocky planets**.

---

## Reference

Lehtmets, A., Kama, M., Fossati, L., & Aret, A. (2025).  
*Investigating Circumstellar Atomic Radiation-driven Dynamics.*  
In revision 2025, DOI: _to be added_.

---

## Repository Structure

```
Betaratio/
├── Main_notebook.ipynb        # Main analysis notebook: beta ratios & velocity boosts
├── spectral_broadening.py     # Example Python script
├── all_elements_masses.txt    # Atomic masses
├── fernandez2006_beta.txt     # Reference comparison data
├── escape_velocity.txt        # Escape velocity data
├── sun_claire2012.dat         # Solar EUV spectrum
├── ... (other input datasets)
└── README.md                  # Project description (this file)
```

---

## Getting Started

### Prerequisites
- Python ≥ 3.8
- Jupyter Notebook
- Required libraries: `numpy`, `matplotlib`, `scipy`, `pandas`, `json`, `os`, `scipy`, `math`

### Installation
```bash
git clone https://github.com/Drannu/Betaratio.git
cd Betaratio
```

### Running the Analysis
To reproduce the calculations from the paper, open:
```bash
jupyter notebook Main_notebook.ipynb
```

---

## Results

- Computed **𝛽 ratios** for atomic species (H → Ni, three ionisation states)  
- Velocity boosts before ionisation  
- Example applications:  
  - Gas-rich debris discs (e.g. β Pictoris, 49 Ceti)  
  - Evaporating hot Jupiters (e.g. HD 209458 b, KELT-9 b)  
  - Rocky planets (e.g. Kepler-1520 b)
  - And much more (e. g. comet tail, ISM, ...)

---

## Cite This Work

If you use this repository or the associated article, please cite:

```bibtex
@article{lehtmets2025beta,
  title={Investigating Circumstellar Atomic Radiation-driven Dynamics},
  author={Lehtmets, A. and Kama, M. and Fossati, L. and Aret, A.},
  year={2025},
  journal={Accepted, DOI pending}
}
```

---

## License

This project is released under the **MIT License**.  
You are free to reuse and adapt the code with proper attribution.

---

## Contact

**Alexandra Lehtmets**  
Tartu Observatory, University of Tartu  
📧 alexandra.lehtmets@ut.ee  
