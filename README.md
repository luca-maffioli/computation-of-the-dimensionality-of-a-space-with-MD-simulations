# Lyapunov spectrum and correlation integral of a sheared 2D WCA fluid

> **Reference.** This code was used to produce the results published in
>
> L. Maffioli, N. Clisby, F. Frascoli and B. D. Todd,
> *The phase space distribution of confined fluids under shear is not fractal*,
> J. Chem. Phys. **154**, 094116 (2021).
> [doi:10.1063/5.0039192](https://doi.org/10.1063/5.0039192)
>
> If you use this code, please cite the paper.

```bibtex
@article{Maffioli2021,
  author  = {Maffioli, Luca and Clisby, Nathan and Frascoli, Federico and Todd, B. D.},
  title   = {The phase space distribution of confined fluids under shear is not fractal},
  journal = {The Journal of Chemical Physics},
  volume  = {154},
  number  = {9},
  pages   = {094116},
  year    = {2021},
  doi     = {10.1063/5.0039192}
}
```

This program is a molecular dynamics (MD) simulation of a small two-dimensional fluid
sheared between two atomistic walls. It can measure one of two things about the chaotic
dynamics:

- **Lyapunov spectrum.** The full spectrum of the whole system (walls + fluid), plus the
  spectrum of the fluid sub-block. It is computed with the Benettin / Gram–Schmidt method,
  by integrating the tangent dynamics alongside the trajectory.
- **Correlation integral.** The Grassberger–Procaccia correlation integral C(r) of the
  fluid phase-space trajectory, used to estimate the correlation dimension of the
  attractor.

The program also accumulates velocity and kinetic-energy profiles across the channel in
both modes.

**Model:**

- Particles interact through the Weeks–Chandler–Andersen (WCA) potential, the
  Lennard-Jones potential cut at r<sub>c</sub> = 2<sup>1/6</sup>σ.
- Wall particles are tethered by harmonic springs to lattice sites. The upper sites move
  at +v<sub>w</sub> and the lower sites at −v<sub>w</sub>.
- Each wall has its own Gaussian thermostat with a proportional feedback term. The fluid
  is not thermostatted directly: it exchanges heat only with the walls.
- The equations of motion are integrated with a 5-value Gear predictor–corrector.

---

## Contents

1. [Building and running](#building-and-running)
2. [Geometry and particle indexing](#geometry-and-particle-indexing)
3. [Equations of motion](#equations-of-motion)
4. [Tangent dynamics and Lyapunov exponents](#tangent-dynamics-and-lyapunov-exponents)
5. [Correlation integral](#correlation-integral)
6. [Parameters](#parameters)
7. [Program flow](#program-flow)
8. [Data structures](#data-structures)
9. [Function reference](#function-reference)
10. [Input and output files](#input-and-output-files)
11. [Notes and limitations](#notes-and-limitations)

---

## Building and running

```bash
g++ -O3 -march=native main.cpp -o shear
echo "0.5 1.0" > parameters.txt      # shear rate, temperature
./shear
```

- The run mode is chosen at compile time with the constants `LYAP` and `CORR`. Set
  exactly one of them to 1.
- The shear rate and temperature are read at run time from `parameters.txt`.
- All output files are written to the working directory.

---

## Geometry and particle indexing

```
        y
        ▲     o   o   o   o        upper wall (NLAYER rows of NSIDE particles)
        │   o   o   o   o           tethered to sites moving at +v_w  →
  +Ly/2 ┼─────────────────────
        │      fluid (NFLUID)       periodic in x, confined in y
  -Ly/2 ┼─────────────────────
        │   o   o   o   o           tethered to sites moving at −v_w  ←
        │     o   o   o   o        lower wall
        └──────────────────────▶ x      (box length Lx)
```

**Periodicity.** Only x is periodic. Distances in x use the minimum-image convention
(`pbcs`). In y the fluid is held in by the walls.

**Walls.**

- Each wall has `NLAYER` rows of `NSIDE` particles, with alternate rows offset by half a
  spacing (triangular packing).
- The lower wall is the mirror image of the upper wall.
- The two walls do not interact with each other.

**Particle indices:**

| Range                  | Particles  |
|------------------------|------------|
| `[0, NWALL)`           | upper wall |
| `[NWALL, 2·NWALL)`     | lower wall |
| `[2·NWALL, NPART)`     | fluid      |

with `NWALL = NLAYER·NSIDE` and `NPART = 2·NWALL + NFLUID`.

**Box size.** With the default values, `rho = NFLUID / (4 r_c²)`, so the box is a square
of side `Lx = Ly = 2 r_c` ≈ 2.245.

---

## Equations of motion

All particles have unit mass.

**Fluid** (no thermostat):

```
dq_i/dt = p_i
dp_i/dt = F_i
```

**Walls** (upper wall shown; the lower wall is the same with −v<sub>w</sub>):

```
dq_i/dt = p_i
dp_i/dt = F_i − α_u (p_i − p_ref)
```

where F<sub>i</sub> includes the WCA forces and the spring force
−K (q<sub>i</sub> − site<sub>i</sub>).

The reference momentum p<sub>ref</sub> depends on `const_velocity`:

| `const_velocity` | p<sub>ref</sub>                                        | F used in α        |
|------------------|--------------------------------------------------------|--------------------|
| 1                | imposed wall velocity (±v<sub>w</sub>, 0)              | F                  |
| 0                | instantaneous mean momentum of the wall ⟨p⟩            | F − ⟨F⟩            |

**Thermostat multiplier:**

```
α = Σ F'·p' / Σ p'·p'  +  Fback · (Σ p'·p' − (2·NWALL − Free_wall)·T) / ((2·NWALL − Free_wall)·T)
```

with p' = p − p<sub>ref</sub>.

- The first term is the Gaussian multiplier: it keeps the peculiar kinetic energy of the
  wall constant.
- The second term pulls that kinetic energy back to its target value.

**Wall sites.** The tether sites move rigidly at ±v<sub>w</sub>, with
v<sub>w</sub> = γ̇ (Ly/2 + σ<sub>W</sub>/2). This is the velocity of the linear shear
profile at the first wall row.

**Integrator.** The equations are integrated with the 5-value Gear predictor–corrector for
first-order equations. The corrector coefficients are 251/720, 1, 11/12, 1/3 and 1/24.
The derivatives are stored scaled: `x1 = dt·dx/dt`, `x2 = dt²/2·d²x/dt²`, …

---

## Tangent dynamics and Lyapunov exponents

**Phase-space vector.** The state is ordered per particle as
(x<sub>0</sub>, y<sub>0</sub>, px<sub>0</sub>, py<sub>0</sub>, x<sub>1</sub>, …), so
particle *i* occupies rows/columns `4i … 4i+3`. The phase space has dimension
`4·NPART`, which is 40 with the default sizes.

**Jacobian.** `flyap` is the Jacobian J of the equations of motion. It is rebuilt every
step from four contributions:

| Block   | Contribution                                                        | Built in          |
|---------|---------------------------------------------------------------------|-------------------|
| dq̇/dp  | identity (unit mass)                                                | `force`           |
| dṗ/dq  | Hessian of the WCA potential                                        | `wcaforce`        |
| dṗ/dq  | −K on the diagonal of the wall particles                            | `springforce`     |
| dṗ/dp, dṗ/dq | thermostat terms: −α ∂p'/∂p, −p' ∂α/∂q, −p' ∂α/∂p           | `thermostatforce` |

**Tangent vectors.** The `4·NPART` tangent vectors are the columns of `L0`. They obey
dL/dt = J·L and are integrated with the same Gear scheme as the trajectory, using
`TT = J·L0` as their time derivative.

**Re-orthonormalisation.** After every step, `orthonormalization` applies Gram–Schmidt to
the vectors, in a metric that weights positions by `FMRAT²` relative to momenta:

```
‖v‖² = Σ_k FMRAT²·(δx_k² + δy_k²) + δpx_k² + δpy_k²
```

Each vector is then rescaled to norm `NLYAP`, and the log of its growth factor
ln(‖v‖/NLYAP) is added to `lyapaverage`.

**Exponents.** The i-th exponent is

```
λ_i = lyapaverage_i / (number of steps · dt)
```

`sortlyap` keeps the exponents (and their vectors) in decreasing order.

**Fluid sub-block.** The same procedure is applied, in parallel, to the fluid block of
the Jacobian alone (`TTF`, `LF0`, `flyapaverage`). This is the spectrum of the fluid with
the wall degrees of freedom held fixed.

---

## Correlation integral

1. **Sampling times.** `TOTSAMPLE` sampling times are drawn after `WARMUP`. Consecutive
   samples are separated by

   ```
   3/4·FREQ  +  an exponential variable with mean FREQ/4
   ```

   so the mean interval is `FREQ` and the minimum interval is 3/4·`FREQ`. The random part
   avoids sampling in step with any periodicity of the dynamics.
2. **Samples.** At each sampling time, the fluid state
   (x, y, px, py for each fluid particle, `dim` = 4·NFLUID values) is stored.
3. **Analysis** (`computecorr`, at the end of the run):
   - every coordinate is divided by its standard deviation;
   - the radii are spaced logarithmically: r<sub>i</sub> = D / 1.125<sup>i</sup>, where D
     is the diagonal of the bounding box of the scaled data (1000 radii, the last one
     being 0);
   - the distances between all pairs of samples are histogrammed and summed cumulatively;
   - the result is normalised by N(N−1)/2:

   ```
   C(r) = (number of pairs closer than r) / (N(N−1)/2)
   ```

The correlation dimension is the slope of log C(r) against log r in the scaling region.

---

## Parameters

### Run mode

| Constant         | Default | Meaning |
|------------------|---------|---------|
| `LYAP`           | 0       | 1 = compute the Lyapunov spectrum |
| `CORR`           | 1       | 1 = compute the correlation integral |
| `const_velocity` | 0       | Thermostat reference: 1 = imposed wall velocity, 0 = instantaneous wall mean momentum |

### Interactions

| Constant                     | Default             | Meaning |
|------------------------------|---------------------|---------|
| `sigmaF`, `sigmaW`           | 1                   | Fluid and wall diameters |
| `epsF`, `epsW`               | 1                   | Fluid and wall energies |
| `sigmaFW`, `epsFW`           | mixing rules        | Fluid–wall parameters (Lorentz–Berthelot) |
| `RFF`, `RWW`, `RFW`          | 2<sup>1/6</sup>σ    | WCA cutoffs |
| `K`                          | 150                 | Spring constant of the wall tethers |
| `Fback`                      | 50                  | Thermostat feedback coefficient |

### System size

| Constant    | Default | Meaning |
|-------------|---------|---------|
| `NFLUID`    | 2       | Fluid particles. Must be 2n², see `setLattice` |
| `NSIDE`     | 2       | Wall particles per row |
| `NLAYER`    | 2       | Rows per wall |
| `NWALL`     | 4       | Particles per wall |
| `NPART`     | 10      | Total number of particles |
| `Free_wall` | 3       | Degrees of freedom constrained per wall: 2 (momentum) + 1 (thermostat) |
| `rho`       | 0.397   | Fluid density, NFLUID / (4 r<sub>c</sub>²) |
| `Lx`, `Ly`  | 2.245   | Box length in x and channel width |

### Time stepping and sampling

| Constant       | Default   | Meaning |
|----------------|-----------|---------|
| `LYAPSTEPSIZE` | 10⁻⁴      | Time step for `LYAP = 1` |
| `CORRSTEPSIZE` | 10⁻³      | Time step for `CORR = 1` |
| `dt`           | —         | `LYAPSTEPSIZE·LYAP + CORRSTEPSIZE·CORR` |
| `WARMUP`       | 10⁹       | Equilibration steps |
| `LYAPSAMPLE`   | 10⁸       | Production steps of the Lyapunov run |
| `SAVEFREQ`     | 10⁵       | Output interval of the Lyapunov run (steps) |
| `FREQ`         | 10⁴       | Mean interval between correlation samples (steps) |
| `TOTSAMPLE`    | 2·10⁶     | Number of correlation samples |
| `dim`          | 8         | Values per correlation sample (= 4·NFLUID) |
| `nbin`, `FRAC` | 1000, 8   | Profile bins over y ∈ [−FRAC/2·Ly, FRAC/2·Ly] |

### Lyapunov algorithm

| Constant         | Default             | Meaning |
|------------------|---------------------|---------|
| `FMRAT`          | 10                  | Weight of positions relative to momenta in the tangent metric |
| `NLYAP`          | 10⁻⁵                | Norm the tangent vectors are rescaled to |
| `kickp`, `kickq` | NLYAP/√2, kickp/(√2·FMRAT) | Initial momentum and position components of the tangent vectors |
| `MINLYAP`        | 10⁻³⁶               | Vectors with a smaller norm are not rescaled |

---

## Program flow

```
main
 ├─ setParameters      zero the state and the profiles, read parameters.txt
 ├─ setLattice         fluid lattice, wall rows, wall tether sites
 ├─ randomMomenta      random momenta at temperature T, plus the shear profile
 ├─ lyapinitialize     diagonal initial tangent basis
 └─ molDyn
     ├─ LYAP = 1: for t = 0 … TOTSTEP
     │     gear
     │     t ≥ WARMUP:  orthonormalization → average → sortlyap
     │                  every SAVEFREQ: write lyap.txt, flyap.txt, check.txt
     │                  every 10 steps: setprofile
     ├─ CORR = 1: generate_times
     │            run gear until TOTSAMPLE samples are stored
     │            t ≥ WARMUP: setprofile every 10 steps
     │            savefiles → computecorr
     └─ write Profiles.txt

gear (one time step)
 ├─ predictor (state, and tangent vectors if t ≥ WARMUP and LYAP = 1)
 ├─ move the wall sites, PBC in x
 ├─ force
 │    ├─ wcaforce          forces (+ Hessian)
 │    ├─ springforce       tether forces (+ −K terms)
 │    ├─ thermostatforce   α_u, α_d (+ thermostat Jacobian terms)
 │    └─ TT = J·L0, TTF = J_fluid·LF0
 ├─ corrector: upper wall, lower wall, fluid, tangent vectors
 └─ PBC in x
```

---

## Data structures

All structures are allocated once in `main`, with sizes fixed at compile time.

### `Phase`: state of the system

| Field                                        | Meaning |
|----------------------------------------------|---------|
| `x0…x4`, `y0…y4`                             | Positions and their scaled derivatives (Gear) |
| `px0…px4`, `py0…py4`                         | Momenta and their scaled derivatives |
| `ux0`, `uy0`                                 | Tether sites of the upper wall |
| `dx0`, `dy0`                                 | Tether sites of the lower ("down") wall |
| `pmxu`, `pmyu`, `pmxd`, `pmyd`               | Mean momentum of the upper and lower wall |
| `adenu`, `adend`                             | Running sums of the thermostat denominators (for `check.txt`) |
| `T`, `gammadot`, `wallvel`                   | Temperature, shear rate, velocity of the wall sites |

### `Interact`: forces and Jacobian

| Field                                        | Meaning |
|----------------------------------------------|---------|
| `fx`, `fy`                                   | Total force on each particle |
| `flyap[4N][4N]`                              | Jacobian J of the equations of motion |
| `TT[4N][4N]`                                 | J·L0 |
| `TTF[4NF][4NF]`                              | J<sub>fluid</sub>·LF0 |
| `fmxu`, `fmyu`, `fmxd`, `fmyd`               | Mean force on each wall |
| `alphau`, `alphad`                           | Thermostat multipliers |
| `avalphau`, `avalphad`                       | Running sums of the multipliers |

### `Lyapforce`: intermediate derivatives

| Field                                        | Meaning |
|----------------------------------------------|---------|
| `dudxx`, `dudxy`, `dudyy`                    | Hessian blocks ∂F/∂q of the WCA potential |
| `dalphaupdx/dy`, `dalphadowndx/dy`           | ∂α/∂q, over wall + fluid positions (size NWALL+NFLUID) |
| `dalphaupdpx/dpy`, `dalphadowndpx/dpy`       | ∂α/∂p, over the momenta of the same wall |

### `Lvec`: tangent vectors and accumulators

| Field                                        | Meaning |
|----------------------------------------------|---------|
| `L0…L4`                                      | Tangent vectors (columns) and their Gear derivatives |
| `LF0…LF4`                                    | Same for the fluid block |
| `lyapexp`, `flyapexp`                        | Log growth factor of each vector over the last step |
| `lyapaverage`, `flyapaverage`                | Running sums of the log growth factors |
| `lyap2average`, `flyap2average`              | Running sums of their squares (for error estimates) |

### `Profile`: profiles across the channel

Each bin in y holds:

- the sum of p<sub>x</sub>;
- the sum of p<sub>x</sub>² + p<sub>y</sub>²;
- the number of samples.

Separate arrays are kept for the fluid (`v`, `Kin`, `nsample`), the upper wall (`vup`,
`Kinup`, `usample`) and the lower wall (`vdown`, `Kindown`, `dsample`).

---

## Function reference

### Initialisation

#### `void setParameters(Phase *phase, Profile *profile)`
Prepares the run:

- zeroes all positions, momenta and their derivatives;
- zeroes the profiles;
- reads `gammadot` and `T` from `parameters.txt`;
- sets `wallvel = gammadot·(Ly/2 + sigmaW/2)`.

#### `void setLattice(Phase *phase)`
Sets the initial positions:

- **Fluid:** a square lattice of side `round(sqrt(NFLUID/2))`, with two particles per
  cell offset by ±¼ of a cell.
- **Upper wall:** `NLAYER` rows of `NSIDE` particles above y = +Ly/2, with spacing
  Lx/NSIDE and rows √3/2 of a spacing apart. Odd rows are offset by half a spacing.
- **Lower wall:** the mirror image below −Ly/2.
- **Tether sites:** start at the initial wall positions.

#### `void randomMomenta(Phase *phase, int min, int max)`
Sets the initial momenta. For each group (fluid, upper wall, lower wall) it:

1. draws random integer momenta in [min, max];
2. removes the mean momentum of the group;
3. rescales to temperature `T`, using 2·NFLUID degrees of freedom for the fluid and
   2·NWALL − Free_wall for each wall;
4. adds the streaming velocity: γ̇·y for the fluid, ±v<sub>w</sub> for the walls.

#### `int randomInteger(int lower, int upper)`
Returns a uniform random integer in [lower, upper], using `rand()`.

#### `void rand0(void)`
Seeds `rand()` with the current time.

#### `void lyapinitialize(Lvec *lvec)`
Prepares the tangent space:

- zeroes all tangent vectors and accumulators;
- sets `L0` (and `LF0`) to a diagonal basis, with `kickq` on the position components and
  `kickp` on the momentum components.

### Driver

#### `void molDyn(Lyapforce *lyap, Phase *phase, Interact *interact, Lvec *lvec, Profile *profile)`
Runs the simulation in the mode selected by `LYAP` / `CORR`. See
[Program flow](#program-flow). It opens `check.txt`, and at the end writes
`Profiles.txt`.

**Lyapunov run (`LYAP = 1`).** Every `SAVEFREQ` steps after `WARMUP`, it writes:

- the running exponents to `lyap.txt` and `flyap.txt`;
- the thermostat diagnostics to `check.txt`.

**Correlation run (`CORR = 1`):**

1. draws the sampling times, and redraws them if the last one falls more than 10 %
   beyond `TOTSTEP`;
2. integrates until all `TOTSAMPLE` fluid states are stored;
3. calls `savefiles` and `computecorr`.

### Integration

#### `void gear(Lyapforce *lyap, Phase *phase, Interact *interact, Lvec *Lvec, long long int t, ofstream &check)`
Advances the system by one time step:

1. **Predictor:** Taylor-expands positions, momenta and their derivatives (Pascal
   triangle update). It does the same for the tangent vectors when `LYAP = 1` and
   `t ≥ WARMUP`.
2. **Wall sites:** moves the upper sites by +v<sub>w</sub>·dt and the lower sites by
   −v<sub>w</sub>·dt.
3. **PBC:** wraps x for the particles and the sites.
4. **Forces:** calls `force` at the predicted state.
5. **Corrector:**
   - upper and lower walls, including the thermostat term α(p − p<sub>ref</sub>);
   - fluid, without a thermostat;
   - tangent vectors, using `TT` and `TTF`.
6. **PBC:** wraps x again.

#### `void pbcs(double *x)`
Wraps an x coordinate, or an x separation, into [−Lx/2, Lx/2).

### Forces and Jacobian

#### `void force(Lyapforce *lyap, Phase *phase, Interact *interact, Lvec *lvec, long long int t, ofstream &check)`
Computes the forces and, when needed, the tangent dynamics:

1. resets the forces. When `LYAP = 1` and `t ≥ WARMUP`, it also resets the Jacobian and
   the derivative arrays;
2. calls `wcaforce`, `springforce` and `thermostatforce`, in that order.
   `thermostatforce` reads the wall rows of the Jacobian built by the first two;
3. when `LYAP = 1` and `t ≥ WARMUP`:
   - adds the identity dq̇/dp block;
   - computes `TT = flyap·L0` and `TTF = flyap_fluid·LF0`. The fluid block starts at
     row/column 8·NWALL.

#### `void wcaforce(Lyapforce *lyap, Phase *phase, Interact *interact, long long int t)`
Computes the WCA forces for all interacting pairs:

- upper wall–upper wall
- lower wall–lower wall
- fluid–fluid
- upper wall–fluid
- lower wall–fluid

The two walls do not interact with each other. For a pair at distance r < r<sub>c</sub>:

```
s  = (σ/r)⁶
g  = 24ε s (2s − 1) / r²          force on i:  F_i = g · r_ij   (r_ij = q_i − q_j)
DF = 24ε s (26s − 7) / r²

∂F_i,a / ∂q_j,b = −g δ_ab + r_a r_b (g + DF) / r²       (a, b ∈ {x, y},  j ≠ i)
∂F_i / ∂q_i     = − Σ_{j≠i} ∂F_i / ∂q_j
```

When `LYAP = 1` and `t ≥ WARMUP`, it also stores the Hessian blocks `dudxx`, `dudxy`,
`dudyy`. The diagonal blocks are minus the sum of the off-diagonal ones, and the result
is added to the dṗ/dq block of `flyap`.

#### `void springforce(Lyapforce *lyap, Phase *phase, Interact *interact, long long int t)`
Adds the tether force −K (q − site) to each wall particle, with the x separation wrapped
by `pbcs`. When `LYAP = 1` and `t ≥ WARMUP`, it adds −K to the corresponding diagonal
entries of `flyap`.

#### `void thermostatforce(Lyapforce *lyap, Phase *phase, Interact *interact, long long int t, ofstream &check)`
Computes the thermostat multipliers α<sub>u</sub> and α<sub>d</sub> (see
[Equations of motion](#equations-of-motion)):

- with `const_velocity = 0`, it first computes the mean momentum and mean force of each
  wall;
- the feedback term is added at the end;
- after `WARMUP`, α and the denominators are added to running sums for `check.txt`.

When `LYAP = 1` and `t ≥ WARMUP`, it also adds the thermostat terms to the Jacobian:

| Term                               | Meaning |
|------------------------------------|---------|
| −α ∂p'/∂p                          | −α on the diagonal. With `const_velocity = 0`, also +α/NWALL on the whole wall block |
| −p' ∂α/∂q                          | ∂α/∂q from the wall rows of `flyap` (WCA + springs), over wall and fluid positions |
| −p' ∂α/∂p                          | Quotient rule on numerator and denominator of α |

The arrays of size NWALL+NFLUID map index *i* to particle *i* for *i* < NWALL, and to
particle *i* + NWALL otherwise, which covers the wall itself and then the fluid.

### Lyapunov exponents

#### `void orthonormalization(Lvec *lvec)`
Applies Gram–Schmidt to the columns of `L0`, then of `LF0`, in the weighted metric (see
[Tangent dynamics](#tangent-dynamics-and-lyapunov-exponents)). Each vector is rescaled to
norm `NLYAP`, and ln(‖v‖/NLYAP) is stored in `lyapexp` (`flyapexp`). Vectors with norm
below `MINLYAP` are not rescaled.

#### `void average(Lvec *lvec)`
Adds `lyapexp` and `lyapexp²` (and the fluid equivalents) to the running sums.

#### `void sortlyap(Lvec *lvec)`
Sorts the exponents in decreasing order of their running sum. It swaps the corresponding
columns of `L0…L4` (`LF0…LF4`), so each vector stays paired with its exponent and
Gram–Schmidt always processes the most unstable direction first.

### Profiles and correlation integral

#### `void setprofile(Phase *phase, Profile *profile, long long int t)`
Bins every particle by its y coordinate, using `nbin` bins over
[−FRAC/2·Ly, FRAC/2·Ly]. For each bin it accumulates p<sub>x</sub>,
p<sub>x</sub>² + p<sub>y</sub>² and the number of samples, separately for the fluid and
each wall. It is called every 10 steps after `WARMUP`.

#### `void generate_times(long long int *v)`
Fills `v` with `TOTSAMPLE` random intervals of 3/4·FREQ plus an exponential variable with
mean FREQ/4. It uses `mt19937_64`, seeded from the clock. `molDyn` turns these intervals
into absolute sampling times.

#### `void computecorr(double *corr)`
Computes the correlation integral of the stored samples (see
[Correlation integral](#correlation-integral)) and writes it to `corr.txt`.

#### `void savefiles(Phase *phase)`
Writes the run parameters to `Info_simulation.txt`.

---

## Input and output files

| File                  | Mode | Content |
|-----------------------|------|---------|
| `parameters.txt`      | in   | `gammadot T` |
| `lyap.txt`            | LYAP | One line every `SAVEFREQ` steps: the 4·NPART running exponents of the full system, in decreasing order, comma-separated |
| `flyap.txt`           | LYAP | Same for the 4·NFLUID exponents of the fluid block |
| `check.txt`           | LYAP | One line every `SAVEFREQ` steps (see below) |
| `corr.txt`            | CORR | 1000 lines of `r C(r)`, from the largest r down to r = 0, with r in units of the standardised coordinates |
| `Info_simulation.txt` | CORR | Run parameters |
| `Profiles.txt`        | both | `nbin` lines (see below) |

**`check.txt` columns:**

1. ⟨α<sub>u</sub>⟩ and ⟨α<sub>d</sub>⟩, averaged since `WARMUP`;
2. the wall mean momenta p<sub>mxu</sub>, p<sub>mxd</sub>, p<sub>myu</sub>,
   p<sub>myd</sub>;
3. the mean thermostat denominators Σp'² of each wall over the last `SAVEFREQ` steps.

**`Profiles.txt` columns:**

```
v  vup  vdown  Kin  Kinup  Kindown  nsample  usample  dsample
```

These are raw sums. Bin *j* is centred at y = (j + ½)·FRAC·Ly/nbin − FRAC/2·Ly. To get
the streaming velocity and kinetic-energy profiles, divide each sum by its sample count.

---

## Notes and limitations

- **One mode at a time.** Set exactly one of `LYAP` and `CORR` to 1. With both at 0,
  `dt = 0`. With both at 1, `dt` is the sum of the two step sizes, and the correlation run
  restarts its step counter after the Lyapunov run.
- **Size assumptions.** The code is written for small systems, with all pairs computed
  directly and no neighbour lists. Some sizes are tied together:
  - `NFLUID` must be 2n² (`setLattice`);
  - `dim` must equal 4·NFLUID;
  - the wall/fluid index mapping in `thermostatforce` assumes NFLUID ≤ NWALL.
- **Feedback term.** It is not included in the Jacobian, so the tangent dynamics follow
  the pure Gaussian thermostat.
- **First step of the Lyapunov run.** The initial tangent vectors have norm NLYAP/2
  (positions) or NLYAP/√2 (momenta) in the weighted metric, not NLYAP. The first
  re-orthonormalisation therefore adds a constant −ln 2 or −ln √2 to the sums, which
  decays as 1/(number of steps).
- **`parameters.txt` missing.** The program prints a warning but continues with
  uninitialised values.
- **Profiles are not bounds-checked.** A particle outside y ∈ [−FRAC/2·Ly, FRAC/2·Ly]
  would write outside the profile arrays.
- **Cost of the correlation integral.** It compares all pairs of samples, so its cost
  grows as TOTSAMPLE²: about 2·10¹² distances with the default settings.
- **Random seeds.** Both generators are seeded from the clock, so every run is
  different. For reproducible runs, replace the seeds in `rand0` and `generate_times`
  with constants.
