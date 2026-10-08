# Virtual Element Method

A small Python implementation of finite element and virtual element constructions on triangular meshes, aimed at demonstrating the reference mapping methodology for virtual element methods that was developed in [VEM_Maps.pdf](VEM_Maps.pdf).

## What this repository does

This repository provides:

- **Classical finite element spaces** on triangles: `C0` Lagrange (`k >= 1`) and Hermite (`k >= 3`) elements of any order.
- **Virtual element spaces** on triangles: the Lagrange-type (`k >= 1`) and Hermite-type (`k >= 3`) spaces of [VEM_Maps.pdf](VEM_Maps.pdf) Section 5 up to order 6, with projections either assembled on each **physical** element or **mapped** from the reference triangle.
- **Assembly routines** for:
  - an **L2 projection** problem
  - a **Poisson** problem with Dirichlet boundary conditions
  - a **heat equation** with theta-scheme timestepping (backward Euler / Crank-Nicolson)
- **Diagnostics** for comparing:
  - mapped vs physical value projectors
  - mapped vs physical gradient projectors
  - approximation errors and convergence rates

The project is intended as a compact implementation of the ideas in the attached paper rather than a full general-purpose VEM library.

## Repository layout

```text
.
├── demo/
│   ├── run_heat.py
│   ├── run_l2_projection.py
│   ├── run_poisson.py
│   └── side_projects/
├── src/
│   └── VEM/
│       ├── spaces/
│       │   ├── fem_space.py
│       │   ├── vem_space.py
│       │   ├── base.py
│       │   └── common/
│       ├── assembly/
│       └── diagnostics/
├── VEM_Maps.pdf
├── pyproject.toml
└── README.md
```

### Main modules

- `src/VEM/spaces/fem_space.py`
  `FEMSpace`: classical finite elements, with the nodal basis built on the reference triangle and mapped to each element.
- `src/VEM/spaces/vem_space.py`
  `VEMSpace`: virtual elements with the constrained least-squares value projection and the gradient projection, physical or reference-mapped.
- `src/VEM/spaces/common/`
  Shared triangle geometry, scaled monomials, the constrained least-squares solver and vertex length scales.
- `src/VEM/assembly/`
  Global assembly routines for the demo problems.
- `src/VEM/diagnostics/`
  Error measures, convergence rates and mapped-vs-physical comparison tools.
- `demo/`
  Small runnable examples showing how to assemble and solve the implemented model problems.

## Installation

The recommended workflow is to install the package in a virtual environment in editable mode.

### 1. Create a virtual environment

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install the package

From the repository root:

```bash
python -m pip install -e .
```

This installs the package in editable mode, so changes under `src/` are picked up without reinstalling.

## Running the demos

Run all demos from the repository root.

### `L2` projection demo

```bash
python demo/run_l2_projection.py
```

This script:
- builds a triangular grid on the unit square,
- constructs one or more FEM/VEM spaces,
- assembles the `L2` projection system,
- solves for the coefficients,
- reports errors,
- and can optionally compare mapped and physical projector implementations and plot.

### Poisson demo

```bash
python demo/run_poisson.py
```

This script:
- builds a sequence of refined triangular meshes,
- assembles a projected Poisson operator,
- applies Dirichlet boundary conditions,
- solves the resulting linear system,
- reports projected `L2` and `H1`-seminorm errors,
- and can optionally compare mapped and physical gradient projectors and plot.

### Heat equation demo

```bash
python demo/run_heat.py
```

This script:
- solves `u_t - Laplace u = f` with the manufactured solution `u = (1 + t) sin(pi x) sin(pi y)`,
- assembles the stiffness matrix and a VEM mass matrix stabilised by `|E| (I - P)^T (I - P)`,
- steps with the theta-scheme (`theta=1/2` Crank-Nicolson, `theta=1` backward Euler), factorising the system once,
- and reports projected `L2` and `H1`-seminorm errors at the final time; the solution is linear in time, so these show the spatial rates.

## Available spaces

Both space classes are exposed through `VEM` and share the same interface (`bind`, `evaluateLocal`, `evaluateLocalGradient`, `interpolate`, `mapper`, `localDofs`):

- `FEMSpace(view, order, element="lagrange")`
- `VEMSpace(view, order, element="lagrange", mapped=False)`

`element` is `"lagrange"` (`order >= 1`) or `"hermite"` (`order >= 3`). `VEMSpace` accepts orders up to `VEMSpace.MAX_ORDER = 6`: beyond that the monomial-based dofs make the value projection lose exactness on `P_k` in double precision. For `VEMSpace`, `evaluateLocal` and `evaluateLocalGradient` return the value projection and gradient projection of the virtual basis, and `mapped=True` evaluates the reference-mapped projections of Section 4.2 instead of assembling them on each element.

```python
from VEM import FEMSpace, VEMSpace

fem = FEMSpace(view, 2)
vem = VEMSpace(view, 4, element="hermite", mapped=True)
```

The demos take a dictionary of named space factories, for example:

```python
from functools import partial

run_poisson_demo(spaces={
    "hermite k=4 FEM": partial(FEMSpace, order=4, element="hermite"),
    "hermite k=4 mapped VEM": partial(VEMSpace, order=4, element="hermite", mapped=True),
})
```

## Notes

- The current implementation is focused on **triangular meshes**.
- The demos use the **DUNE Python bindings** and `aluConformGrid`.
- The code is intended as a readable implementation/prototype accompanying the theory note.

## License

Released under the MIT License. See `LICENSE`.
