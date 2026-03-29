# Virtual Element Method

A small Python implementation of finite element and virtual element constructions on triangular meshes, aimed at demonstrating the reference mapping methodology for virtual element methods.

## What this repository does

This repository provides:

- **Classical finite element spaces** on triangles:
  - linear Lagrange
  - quadratic Lagrange
  - cubic Hermite
- **Virtual element spaces** on triangles:
  - linear Lagrange-type VEM
  - cubic Hermite-type VEM
  - both **physical** and **mapped/reference-based** variants
- **Assembly routines** for:
  - an **L2 projection** problem
  - a **Poisson** problem with Dirichlet boundary conditions
- **Diagnostics** for comparing:
  - mapped vs physical value projectors
  - mapped vs physical gradient projectors
  - approximation errors

The project is intended as a compact implementation of the ideas in the attached paper rather than a full general-purpose VEM library.

## Repository layout

```text
.
├── demo/
│   ├── run_l2_projection.py
│   └── run_poisson.py
├── src/
│   └── VEM/
│       ├── assembly/
│       ├── diagnostics/
│       └── spaces/
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Main modules

- `src/VEM/spaces/`
  Finite element and virtual element space definitions.
- `src/VEM/assembly/`
  Global assembly routines for the demo problems.
- `src/VEM/diagnostics/`
  Error measures and mapped-vs-physical comparison tools.
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
- constructs one or more finite/VEM spaces,
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

## Available spaces

The package currently exposes the following space families through `VEM`:

- `LinearLagrangeSpace`
- `QuadraticLagrangeSpace`
- `CubicHermiteSpace`
- `LinearLagrangeMappedVEMSpace`
- `LinearLagrangePhysicalVEMSpace`
- `CubicHermiteMappedVEMSpace`
- `CubicHermitePhysicalVEMSpace`

## Notes

- The current implementation is focused on **triangular meshes**.
- The demos use the **DUNE Python bindings** and `aluConformGrid`.
- The code is intended as a readable implementation/prototype accompanying the theory note.

## License

Released under the MIT License. See `LICENSE`.