# pyvpsc
<<<<<<< HEAD

Strain-rate driven viscoplastic self-consistent (VPSC) polycrystal model, built on `pyhom`
(Hill tensor, `Phase` / `Aggregate`). Fixed texture, no hardening.

    parameters.py          <- the file you edit (texture, crystal, CRSS, gamma0, n, D, scheme)
    run.py                 <- python run.py
    textures/              <- texture files (VPSC format: shape line, then phi1 Phi phi2 weight)
    docs/EQUATIONS.md      <- every equation, numbered, with the line of code that implements it
    src/pyvpsc/
        texture.py         read_texture, Bunge Euler -> rotation
        slip_systems.py    FCC / BCC / HCP libraries, active set with CRSS
        grain_law.py       power law, grain Newton solver
        linearization.py   secant, affine (explicit formulas)
        vpsc.py            the self-consistent algorithm, step by step + check_equations
        taylor.py          Taylor model (also the starting point of the iteration)
        aggregate.py       builds the pyhom Aggregate (one Phase per grain)

Slip systems: each individual system has its own `active`, `tau_c` and `gamma0`. Set them per family
(`SLIP` in `parameters.py`), then override single systems with `SYSTEM_OVERRIDES = {("prismatic", 2): dict(active=False)}`.
`python run.py --systems` lists every system with its index and Miller(-Bravais) label.

Install (pyhom first) and run:

    python -m pip install -e ../pyhom
    python -m pip install -e ".[dev]"
    python -m pytest
    python run.py

Conventions: Mandel 6-vectors / 6x6 matrices (xx, yy, zz, xy, xz, yz); Bunge angles in degrees,
g = sample -> crystal, R = g^T = crystal -> sample; HCP crystal frame: z // [0001], x // [2-1-10].

>>>>>>> 95304d780b5a2c98f493215198420d03d0d1d828
