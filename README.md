# MATMEK-4270 Mandatory Assignment 1

[![MAT-MEK4270 mandatory 1](https://github.com/FrancescoMinisini/mandatory1/workflows/MAT-MEK4270%20mandatory%201/badge.svg)](https://github.com/FrancescoMinisini/mandatory1/actions)

This repository is my fork for the first mandatory assignment in MATMEK-4270 (Numerical Methods for PDEs). It includes implementations of 2D Poisson and wave equation solvers (Dirichlet and Neumann boundaries), convergence tests, a report with analytical derivations, and an animation of the Neumann wave problem.

## Key Files
- `Poisson2D.py`: 2D Poisson solver with manufactured solutions.
- `Wave2D.py`: 2D wave solver (Neumann and Dirichlet BCs).
- `plot.py` : Python Module that generates the `neumannwave.gif` file inside the `report/` folder
- `report/`: Notebook report with answers 1.2.3 , 1.2.4 and 1.2.6 including the `neumannwave.gif` animation.
---
## Generate Animation

I added a new python module dedicated to the creation of the .gif file. The parameters of the animation can be adjusted inside the plot.py module. To create the animatin, run the command

```bash
    python plot.py
```

Run tests with `pytest Wave2d.py`. Due date: October 10, 2025.