/*
 * ============================================================================
 *  2D WCA fluid sheared between two thermostatted walls
 *  - Lyapunov spectrum (full system and fluid-only sub-block)
 *  - Correlation integral of the fluid phase-space trajectory
 * ============================================================================
 *
 *  Geometry
 *    - x is periodic (box length Lx). In y the fluid is confined between two
 *      walls (no periodicity in y).
 *    - Each wall has NLAYER layers of NSIDE particles. Every wall particle is
 *      tethered by a harmonic spring (constant K) to a lattice site. The
 *      lattice sites move at +wallvel (upper wall) and -wallvel (lower wall),
 *      which shears the fluid.
 *    - The two walls are thermostatted separately with a Gaussian
 *      thermostat plus a proportional feedback term. The fluid is not
 *      thermostatted directly.
 *
 *  Particle indexing
 *    [0, NWALL)           upper wall
 *    [NWALL, 2*NWALL)     lower wall
 *    [2*NWALL, NPART)     fluid
 *
 *  Run modes (set at compile time)
 *    LYAP = 1 : evolve the tangent-space vectors together with the trajectory
 *               and accumulate the Lyapunov exponents (Benettin / Gram-Schmidt)
 *    CORR = 1 : store the fluid phase-space point at random times and compute
 *               the Grassberger-Procaccia correlation integral at the end
 *
 *  Input : parameters.txt   ->  shear rate (gammadot)  temperature (T)
 *  Output: see molDyn() and computecorr()
 * ============================================================================
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>      // mt19937_64, seed_seq, uniform_real_distribution (generate_times)
#include <sstream>
#include <string>
#include <vector>

using namespace std;


/* ============================================================================
 *  Run mode
 * ========================================================================== */

const long long int LYAP = 0;            // 1 = compute the Lyapunov exponents, 0 = do not
const long long int CORR = 1;            // 1 = compute the correlation integral, 0 = do not
const long long int const_velocity = 0;  // wall thermostat reference velocity:
                                         //   1 = the imposed wall velocity (+/- wallvel)
                                         //   0 = the instantaneous mean velocity of the wall, recomputed every step


/* ============================================================================
 *  Interaction parameters (WCA = LJ cut at its minimum, 2^(1/6) sigma)
 * ========================================================================== */

const double sigmaF = 1;  // fluid diameter
const double sigmaW = 1;  // wall diameter

const double epsF = 1;    // fluid-fluid energy
const double epsW = 1;    // wall-wall energy

const double sigmaFW = (sigmaF + sigmaW) / 2;  // Lorentz-Berthelot mixing rules
const double epsFW   = pow(epsF * epsW, 0.5);

const double RFF = pow(2, 1. / 6.) * sigmaF;   // cutoffs
const double RWW = pow(2, 1. / 6.) * sigmaW;
const double RFW = pow(2, 1. / 6.) * sigmaFW;

const double R2F  = RFF * RFF;                 // squared cutoffs
const double R2W  = RWW * RWW;
const double R2FW = RFW * RFW;


/* ============================================================================
 *  System size
 * ========================================================================== */

const int NFLUID = 2;  // fluid particles (must be 2*n^2, see setLattice)
const int NSIDE  = 2;  // wall particles per layer
const int NLAYER = 2;  // layers per wall

const double rho   = double(NFLUID) / (4 * RFF * RFF);  // fluid density
const double Fback = 50;   // feedback coefficient of the wall thermostats
const double K     = 150;  // spring constant tethering wall particles to their sites

const int NWALL     = NLAYER * NSIDE;      // particles per wall
const int NPART     = 2 * NWALL + NFLUID;  // total number of particles
const int Free_wall = 3;                   // constrained dof per wall (2 momenta + 1 thermostat)

const double Volume = ((double)NFLUID) / rho;
const double Lx  = max(2 * RFF, pow(Volume, 0.5));  // box length in x (periodic)
const double Ly  = Volume / Lx;                     // channel width in y
const double Lx2 = Lx / 2;
const double Ly2 = Ly / 2;


/* ============================================================================
 *  Time stepping and sampling
 * ========================================================================== */

const double LYAPSTEPSIZE = 0.0001;  // time step used when LYAP = 1
const double CORRSTEPSIZE = 0.001;   // time step used when CORR = 1
const double dt = LYAPSTEPSIZE * LYAP + CORRSTEPSIZE * CORR;

const long long int WARMUP     = 1000000000;  // equilibration steps
const long long int FREQ       = 10000;       // mean number of steps between two correlation samples
const long long int TOTSAMPLE  = 2000000;     // number of correlation samples
const long long int LYAPSAMPLE = 100000000;   // production steps of the Lyapunov run

const long long int TOTSTEP = WARMUP + LYAPSAMPLE * LYAP + FREQ * TOTSAMPLE * CORR;

const long long int SAVEFREQ = 100000;  // output interval of the Lyapunov run (steps)

const int dim = 8;                   // dimension of a correlation sample: (x, y, px, py) of each fluid particle = 4*NFLUID
const long long int N = TOTSAMPLE;   // number of correlation samples

const int nbin = 1000;  // bins of the velocity/kinetic-energy profiles
const int FRAC = 8;     // the profiles span y in [-FRAC/2*Ly, FRAC/2*Ly]


/* ============================================================================
 *  Gear predictor-corrector (first-order equations) and Lyapunov parameters
 * ========================================================================== */

const double f01 = 251.0 / 720.0;
const double f21 = 11.0 / 12.0;
const double f31 = 1.0 / 3.0;
const double f41 = 1.0 / 24.0;

const double FMRAT  = 10;              // weight of positions relative to momenta in the tangent-space metric
const double FMRAT2 = FMRAT * FMRAT;
const double NLYAP  = 0.00001;         // norm to which every tangent vector is renormalised
const double kickp  = NLYAP / sqrt(2.);          // initial momentum component of the tangent vectors
const double kickq  = kickp / FMRAT / sqrt(2.);  // initial position component of the tangent vectors

const long double MINLYAP = 0.000000000000000000000000000000000001;  // below this norm a vector is not renormalised


/* ============================================================================
 *  Data structures
 * ========================================================================== */

// Phase-space state. Positions and momenta with their scaled time derivatives
// up to 4th order (Gear), wall lattice sites and wall thermostat quantities.
typedef struct
{
    double x0[NPART], x1[NPART], x2[NPART], x3[NPART], x4[NPART];
    double y0[NPART], y1[NPART], y2[NPART], y3[NPART], y4[NPART];

    double px0[NPART], px1[NPART], px2[NPART], px3[NPART], px4[NPART];
    double py0[NPART], py1[NPART], py2[NPART], py3[NPART], py4[NPART];

    double ux0[NWALL];  // x of the lattice sites of the upper wall
    double uy0[NWALL];  // y of the lattice sites of the upper wall
    double dx0[NWALL];  // x of the lattice sites of the lower ("down") wall
    double dy0[NWALL];  // y of the lattice sites of the lower wall

    double pmxu, pmyu;  // mean momentum of the upper wall
    double pmxd, pmyd;  // mean momentum of the lower wall

    double adenu;       // sum over steps of the thermostat denominator (upper wall), for check.txt
    double adend;       // same, lower wall

    double T;           // temperature (from parameters.txt)
    double gammadot;    // shear rate (from parameters.txt)
    double wallvel;     // velocity of the wall lattice sites
} Phase;

// Forces, Jacobian of the equations of motion and thermostat multipliers
typedef struct
{
    double fx[NPART];
    double fy[NPART];

    double flyap[4 * NPART][4 * NPART];  // Jacobian of (dq/dt, dp/dt) w.r.t. (q, p); row/column 4i..4i+3 = x, y, px, py of particle i
    double TT[4 * NPART][4 * NPART];     // flyap * L0 (time derivative of the tangent vectors)
    double TTF[4 * NFLUID][4 * NFLUID];  // same for the fluid-only block

    double fmxu, fmyu;  // mean force on the upper wall
    double fmxd, fmyd;  // mean force on the lower wall

    double alphau;      // thermostat multiplier, upper wall
    double alphad;      // thermostat multiplier, lower wall

    double avalphau;    // running sums of the multipliers (after WARMUP)
    double avalphad;
} Interact;

// Velocity and kinetic-energy profiles across the channel (raw sums)
typedef struct
{
    long double v[nbin];      // fluid: sum of px
    long double Kin[nbin];    // fluid: sum of px^2 + py^2

    long double vdown[nbin];  // lower wall
    long double Kindown[nbin];

    long double vup[nbin];    // upper wall
    long double Kinup[nbin];

    long long int nsample[nbin];  // number of fluid samples per bin
    long long int usample[nbin];  // upper wall
    long long int dsample[nbin];  // lower wall
} Profile;

// Derivatives needed to build the Jacobian
typedef struct
{
    // Hessian blocks of the WCA potential
    double dudxx[NPART][NPART];
    double dudxy[NPART][NPART];
    double dudyy[NPART][NPART];

    // derivatives of the thermostat multipliers w.r.t. positions (walls + fluid)
    double dalphaupdx[NWALL + NFLUID];
    double dalphaupdy[NWALL + NFLUID];
    double dalphadowndx[NWALL + NFLUID];
    double dalphadowndy[NWALL + NFLUID];

    // derivatives of the thermostat multipliers w.r.t. the wall momenta
    double dalphaupdpx[NWALL];
    double dalphaupdpy[NWALL];
    double dalphadowndpx[NWALL];
    double dalphadowndpy[NWALL];
} Lyapforce;

// Tangent vectors (columns) with their Gear derivatives, and Lyapunov accumulators
typedef struct
{
    double L0[4 * NPART][4 * NPART];
    double L1[4 * NPART][4 * NPART];
    double L2[4 * NPART][4 * NPART];
    double L3[4 * NPART][4 * NPART];
    double L4[4 * NPART][4 * NPART];

    double LF0[4 * NFLUID][4 * NFLUID];  // fluid-only block
    double LF1[4 * NFLUID][4 * NFLUID];
    double LF2[4 * NFLUID][4 * NFLUID];
    double LF3[4 * NFLUID][4 * NFLUID];
    double LF4[4 * NFLUID][4 * NFLUID];

    double lyapexp[4 * NPART];       // log expansion of each vector in the last step
    double lyapaverage[4 * NPART];   // running sum of lyapexp
    double lyap2average[4 * NPART];  // running sum of lyapexp^2

    double flyapexp[4 * NPART];      // same for the fluid block (only 4*NFLUID entries used)
    double flyapaverage[4 * NPART];
    double flyap2average[4 * NPART];
} Lvec;


/* ============================================================================
 *  Function declarations
 * ========================================================================== */

void setParameters(Phase *phase, Profile *profile);
void setLattice(Phase *phase);
void randomMomenta(Phase *phase, int min, int max);
int  randomInteger(int lower, int upper);
void rand0(void);
void molDyn(Lyapforce *lyap, Phase *phase, Interact *interact, Lvec *Lvec, Profile *profile);
void gear(Lyapforce *lyap, Phase *phase, Interact *interact, Lvec *Lvec, long long int t, ofstream &check);
void pbcs(double *x);
void savefiles(Phase *phase);
void force(Lyapforce *lyap, Phase *phase, Interact *interact, Lvec *lvec, long long int t, ofstream &check);
void orthonormalization(Lvec *lvec);
void lyapinitialize(Lvec *lvec);
void average(Lvec *lvec);
void sortlyap(Lvec *lvec);
void setprofile(Phase *phase, Profile *profile, long long int t);
void thermostatforce(Lyapforce *lyap, Phase *phase, Interact *interact, long long int t, ofstream &check);
void wcaforce(Lyapforce *lyap, Phase *phase, Interact *interact, long long int t);
void springforce(Lyapforce *lyap, Phase *phase, Interact *interact, long long int t);
void generate_times(long long int *v);
void computecorr(double *corr);


/* ============================================================================
 *  Main
 * ========================================================================== */

int main()
{
    Phase     *phase    = new Phase;
    Interact  *interact = new Interact;
    Lvec      *lvec     = new Lvec;
    Profile   *profile  = new Profile;
    Lyapforce *lyap     = new Lyapforce;

    setParameters(phase, profile);   // read gammadot and T, zero the state and the profiles
    setLattice(phase);               // initial positions of fluid and walls
    randomMomenta(phase, 0, 100);    // initial momenta at temperature T with the shear profile
    lyapinitialize(lvec);            // initial tangent vectors

    molDyn(lyap, phase, interact, lvec, profile);

    return 0;
}


/* ============================================================================
 *  Initialisation
 * ========================================================================== */

// Zero positions, momenta and their derivatives, zero the profiles, and read
// the shear rate and the temperature from parameters.txt.
void setParameters(Phase *phase, Profile *profile)
{
    int i;
    for (i = 0; i < NPART; i++)
    {
        phase->x0[i] = 0.;
        phase->x1[i] = 0.;
        phase->x2[i] = 0.;
        phase->x3[i] = 0.;
        phase->x4[i] = 0.;
        phase->px0[i] = 0.;
        phase->px1[i] = 0.;
        phase->px2[i] = 0.;
        phase->px3[i] = 0.;
        phase->px4[i] = 0.;
        phase->y0[i] = 0.;
        phase->y1[i] = 0.;
        phase->y2[i] = 0.;
        phase->y3[i] = 0.;
        phase->y4[i] = 0.;
        phase->py0[i] = 0.;
        phase->py1[i] = 0.;
        phase->py2[i] = 0.;
        phase->py3[i] = 0.;
        phase->py4[i] = 0.;
    }

    for (i = 0; i < nbin; i++)
    {
        profile->v[i] = 0;
        profile->Kin[i] = 0;
        profile->nsample[i] = 0;

        profile->vup[i] = 0;
        profile->Kinup[i] = 0;
        profile->usample[i] = 0;

        profile->vdown[i] = 0;
        profile->Kindown[i] = 0;
        profile->dsample[i] = 0;
    }

    // parameters.txt: gammadot T
    ifstream params("parameters.txt");
    if (!params.is_open())
    {
        cout << "There was a problem opening the input file\n";
    }
    params >> phase->gammadot;
    params >> phase->T;
    params.close();

    // velocity of the wall sites: shear profile evaluated at the first wall layer
    phase->wallvel = phase->gammadot * (Ly2 + sigmaW / 2);
}

// Initial positions.
// Fluid: square lattice of side `lattice` with 2 particles per cell.
// Walls: NLAYER staggered (triangular) rows of NSIDE particles above +Ly/2
// (upper wall) and mirrored below -Ly/2 (lower wall). The lattice sites start
// at the initial wall positions.
void setLattice(Phase *phase)
{
    int i, j;

    // fluid
    int c = 2 * NWALL;
    int lattice = round(sqrt(NFLUID / 2));

    double lengx = Lx / lattice;
    double lengy = Ly / lattice;

    double posx, posy;

    for (i = 0; i < lattice; i++)
    {
        for (j = 0; j < lattice; j++)
        {
            posx = (i + 0.5) * lengx;
            posy = (j + 0.5) * lengy;

            phase->x0[c] = posx + lengx / 4 - Lx / 2;
            phase->y0[c] = posy + lengy / 4 - Ly / 2;
            c++;

            phase->x0[c] = posx - lengx / 4 - Lx / 2;
            phase->y0[c] = posy - lengy / 4 - Ly / 2;
            c++;
        }
    }

    // walls: particle i + j*NSIDE is the i-th particle of layer j
    lengx = Lx * (1. / NSIDE);

    for (i = 0; i < NSIDE; i++)
    {
        for (j = 0; j < NLAYER; j++)
        {
            phase->x0[i + j * NSIDE] = (i + 0.5 + 0.5 * (j % 2)) * lengx - Lx / 2;   // odd layers shifted by half a spacing
            phase->x0[i + j * NSIDE + NWALL] = phase->x0[i + j * NSIDE];

            phase->y0[i + j * NSIDE] = lengx * (0.5 + j * sqrt(3) / 2);              // triangular layer spacing
            phase->y0[i + j * NSIDE + NWALL] = -phase->y0[i + j * NSIDE];

            phase->y0[i + j * NSIDE] += Ly / 2;
            phase->y0[i + j * NSIDE + NWALL] -= Ly / 2;
        }
    }

    // lattice sites of the wall springs
    for (i = 0; i < NWALL; i++)
    {
        phase->ux0[i] = phase->x0[i];
        phase->uy0[i] = phase->y0[i];

        phase->dx0[i] = phase->x0[i + NWALL];
        phase->dy0[i] = phase->y0[i + NWALL];
    }
}

// Initial momenta, drawn as random integers in [min, max] and treated
// separately for the fluid and the two walls:
//   1. remove the mean momentum of the group
//   2. rescale to the target temperature
//   3. add the streaming velocity (fluid: gammadot*y; walls: +/- wallvel)
void randomMomenta(Phase *phase, int min, int max)
{
    double pxsum = 0, pysum = 0, kineticTemp = 0, scale;
    int i;

    rand0();

    // --- fluid ---
    pxsum = 0;
    pysum = 0;
    kineticTemp = 0;
    for (i = 2 * NWALL; i < NPART; i++)
    {
        phase->px0[i] = (double)randomInteger(min, max);
        phase->py0[i] = (double)randomInteger(min, max);
    }

    for (i = 2 * NWALL; i < NPART; i++)
    {
        pxsum += phase->px0[i];
        pysum += phase->py0[i];
    }

    pxsum /= NFLUID;
    pysum /= NFLUID;

    for (i = 2 * NWALL; i < NPART; i++)
    {
        phase->px0[i] -= pxsum;
        phase->py0[i] -= pysum;
    }

    for (i = 2 * NWALL; i < NPART; i++)
    {
        kineticTemp += phase->px0[i] * phase->px0[i] + phase->py0[i] * phase->py0[i];
    }

    scale = sqrt((2 * NFLUID) * phase->T / kineticTemp);

    for (i = 2 * NWALL; i < NPART; i++)
    {
        phase->px0[i] *= scale;
        phase->py0[i] *= scale;

        phase->px0[i] += phase->gammadot * phase->y0[i];   // linear shear profile
    }

    // --- upper wall ---
    pxsum = 0;
    pysum = 0;
    kineticTemp = 0;
    for (i = 0; i < NWALL; i++)
    {
        phase->px0[i] = (double)randomInteger(min, max);
        phase->py0[i] = (double)randomInteger(min, max);
    }

    for (i = 0; i < NWALL; i++)
    {
        pxsum += phase->px0[i];
        pysum += phase->py0[i];
    }

    pxsum /= NWALL;
    pysum /= NWALL;

    for (i = 0; i < NWALL; i++)
    {
        phase->px0[i] -= pxsum;
        phase->py0[i] -= pysum;
    }

    for (i = 0; i < NWALL; i++)
    {
        kineticTemp += phase->px0[i] * phase->px0[i] + phase->py0[i] * phase->py0[i];
    }

    scale = sqrt((2 * NWALL - Free_wall) * phase->T / kineticTemp);

    for (i = 0; i < NWALL; i++)
    {
        phase->px0[i] *= scale;
        phase->py0[i] *= scale;

        phase->px0[i] += phase->wallvel;
    }

    // --- lower wall ---
    pxsum = 0;
    pysum = 0;
    kineticTemp = 0;

    for (i = NWALL; i < 2 * NWALL; i++)
    {
        phase->px0[i] = (double)randomInteger(min, max);
        phase->py0[i] = (double)randomInteger(min, max);
    }

    for (i = NWALL; i < 2 * NWALL; i++)
    {
        pxsum += phase->px0[i];
        pysum += phase->py0[i];
    }

    pxsum /= NWALL;
    pysum /= NWALL;

    for (i = NWALL; i < 2 * NWALL; i++)
    {
        phase->px0[i] -= pxsum;
        phase->py0[i] -= pysum;
    }

    for (i = NWALL; i < 2 * NWALL; i++)
    {
        kineticTemp += phase->px0[i] * phase->px0[i] + phase->py0[i] * phase->py0[i];
    }

    scale = sqrt((2 * NWALL - Free_wall) * phase->T / kineticTemp);

    for (i = NWALL; i < 2 * NWALL; i++)
    {
        phase->px0[i] *= scale;
        phase->py0[i] *= scale;

        phase->px0[i] += -phase->wallvel;
    }
}

// Uniform random integer in [lower, upper]
int randomInteger(int lower, int upper)
{
    double f;
    f = (double)rand() / ((double)RAND_MAX + 1);
    return (lower + (int)(f * (upper - lower + 1)));
}

// Seed rand() with the current time
void rand0(void)
{
    srand((int) time (NULL));
}


/* ============================================================================
 *  Driver
 * ========================================================================== */

// Main loop.
//
// LYAP = 1: integrate TOTSTEP steps. After WARMUP, every step: Gram-Schmidt
//   re-orthonormalisation, accumulation and sorting of the exponents. Every
//   SAVEFREQ steps writes
//     lyap.txt  : running Lyapunov exponents of the full system
//     flyap.txt : running Lyapunov exponents of the fluid block
//     check.txt : mean thermostat multipliers, wall mean momenta, mean
//                 thermostat denominators over the last SAVEFREQ steps
//
// CORR = 1: draw TOTSAMPLE random sampling times after WARMUP, integrate until
//   all samples are taken, store (x, y, px, py) of the fluid at each sampling
//   time, then compute the correlation integral (corr.txt).
//
// In both modes the profiles are sampled every 10 steps after WARMUP and
// written to Profiles.txt at the end.
void molDyn(Lyapforce *lyap, Phase *phase, Interact *interact, Lvec *lvec, Profile *profile)
{
    long long int i, j;

    interact->avalphau = 0;
    interact->avalphad = 0;
    phase->adenu = 0;
    phase->adend = 0;

    ofstream check("check.txt");

    // ------------------------------------------------------------ Lyapunov run
    if (LYAP == 1)
    {
        ofstream lex("lyap.txt");
        ofstream flex("flyap.txt");
        for (i = 0; i < TOTSTEP; i++)
        {
            gear(lyap, phase, interact, lvec, i, check);

            if (i >= WARMUP)
            {
                orthonormalization(lvec);   // renormalise the tangent vectors, get the log expansions
                average(lvec);              // accumulate them
                sortlyap(lvec);             // keep the vectors ordered by decreasing exponent

                if (i % SAVEFREQ == 0)
                {
                    for (j = 0; j < 4 * NPART; j++)
                    {
                        lex << lvec->lyapaverage[j] / ((double)(i + 1 - WARMUP) * dt) << ",";
                    }
                    lex << endl;
                    for (j = 0; j < 4 * NFLUID; j++)
                    {
                        flex << lvec->flyapaverage[j] / ((double)(i + 1 - WARMUP) * dt) << ",";
                    }
                    flex << endl;

                    check << interact->avalphau / ((double)(i + 1 - WARMUP)) << "," << interact->avalphad / ((double)(i + 1 - WARMUP)) << ",";
                    check << phase->pmxu << "," << phase->pmxd << "," << phase->pmyu << "," << phase->pmyd << ",";
                    check << phase->adenu / SAVEFREQ << "," << phase->adend / SAVEFREQ << endl;
                    phase->adenu = 0;
                    phase->adend = 0;
                }
            }

            if (i >= WARMUP && i % 10 == 0)
            {
                setprofile(phase, profile, i);
            }
        }
        lex.close();
        flex.close();
    }

    // ------------------------------------------------- correlation-integral run
    if (CORR == 1)
    {
        double *corr = new double[dim * N];          // sample c is corr[dim*c .. dim*c+dim-1]
        long long int c = 0;                         // samples taken so far
        long long int *vtime = new long long int[TOTSAMPLE];

        // Sampling times: WARMUP + cumulative sum of random intervals.
        // Redraw if the last time is more than 10% beyond TOTSTEP.
        do
        {
            generate_times(vtime);

            vtime[0] += WARMUP;

            for (i = 1; i < TOTSAMPLE; i++)
            {
                vtime[i] += vtime[i - 1];
            }
        }
        while (vtime[TOTSAMPLE - 1] > 1.1 * TOTSTEP);

        i = 0;

        while (c < TOTSAMPLE)
        {
            gear(lyap, phase, interact, lvec, i, check);

            // store the fluid phase-space point at the sampling times
            if (i == vtime[c])
            {
                for (j = 0; j < NFLUID; j++)
                {
                    corr[dim * c + j * 4]     = phase->x0[2 * NWALL + j];
                    corr[dim * c + j * 4 + 1] = phase->y0[2 * NWALL + j];
                    corr[dim * c + j * 4 + 2] = phase->px0[2 * NWALL + j];
                    corr[dim * c + j * 4 + 3] = phase->py0[2 * NWALL + j];
                }
                c++;
            }

            if (i >= WARMUP && i % 10 == 0)
            {
                setprofile(phase, profile, i);
            }
            i++;
        }

        savefiles(phase);     // Info_simulation.txt
        computecorr(corr);    // corr.txt

        delete[] corr;
    }

    // ------------------------------------------------------------------ profiles
    // Profiles.txt, one line per bin:
    // v  vup  vdown  Kin  Kinup  Kindown  nsample  usample  dsample
    ofstream input("Profiles.txt");
    for (j = 0; j < nbin; j++)
    {
        input << profile->v[j] << " ";
        input << profile->vup[j] << " ";
        input << profile->vdown[j] << " ";

        input << profile->Kin[j] << " ";
        input << profile->Kinup[j] << " ";
        input << profile->Kindown[j] << " ";

        input << profile->nsample[j] << " ";
        input << profile->usample[j] << " ";
        input << profile->dsample[j] << endl;
    }
    input.close();

    check.close();
}


/* ============================================================================
 *  Integrator
 * ========================================================================== */

// One time step of the 5-value Gear predictor-corrector.
// Equations of motion (unit mass):
//   dq/dt = p
//   dp/dt = F - alpha*(p - p_ref)     (walls, with p_ref set by const_velocity)
//   dp/dt = F                         (fluid)
// When LYAP = 1 and t >= WARMUP, the tangent vectors L (and LF) are integrated
// with the same scheme: dL/dt = J*L, with the Jacobian J built in force().
void gear(Lyapforce *lyap, Phase *phase, Interact *interact, Lvec *Lvec, long long int t, ofstream &check)
{
    int i, j;
    double xcor, ycor, pxcor, pycor, thermx, thermy;

    // ---------------------------------------------------------------- predictor
    for (i = 0; i < NPART; i++)
    {
        phase->x0[i] += phase->x1[i] + phase->x2[i] + phase->x3[i] + phase->x4[i];
        phase->y0[i] += phase->y1[i] + phase->y2[i] + phase->y3[i] + phase->y4[i];

        phase->x1[i] += 2.0 * phase->x2[i] + 3.0 * phase->x3[i] + 4.0 * phase->x4[i];
        phase->y1[i] += 2.0 * phase->y2[i] + 3.0 * phase->y3[i] + 4.0 * phase->y4[i];

        phase->x2[i] += 3.0 * phase->x3[i] + 6.0 * phase->x4[i];
        phase->y2[i] += 3.0 * phase->y3[i] + 6.0 * phase->y4[i];

        phase->x3[i] += 4.0 * phase->x4[i];
        phase->y3[i] += 4.0 * phase->y4[i];

        phase->px0[i] += phase->px1[i] + phase->px2[i] + phase->px3[i] + phase->px4[i];
        phase->py0[i] += phase->py1[i] + phase->py2[i] + phase->py3[i] + phase->py4[i];

        phase->px1[i] += 2.0 * phase->px2[i] + 3.0 * phase->px3[i] + 4.0 * phase->px4[i];
        phase->py1[i] += 2.0 * phase->py2[i] + 3.0 * phase->py3[i] + 4.0 * phase->py4[i];

        phase->px2[i] += 3.0 * phase->px3[i] + 6.0 * phase->px4[i];
        phase->py2[i] += 3.0 * phase->py3[i] + 6.0 * phase->py4[i];

        phase->px3[i] += 4.0 * phase->px4[i];
        phase->py3[i] += 4.0 * phase->py4[i];
    }

    // predictor for the tangent vectors
    if (LYAP == 1 && t >= WARMUP)
    {
        for (i = 0; i < 4 * NPART; i++)
        {
            for (j = 0; j < 4 * NPART; j++)
            {
                Lvec->L0[i][j] +=     Lvec->L1[i][j] +     Lvec->L2[i][j] +     Lvec->L3[i][j] + Lvec->L4[i][j];
                Lvec->L1[i][j] += 2.0*Lvec->L2[i][j] + 3.0*Lvec->L3[i][j] + 4.0*Lvec->L4[i][j];
                Lvec->L2[i][j] += 3.0*Lvec->L3[i][j] + 6.0*Lvec->L4[i][j];
                Lvec->L3[i][j] += 4.0*Lvec->L4[i][j];
            }
        }

        for (i = 0; i < 4 * NFLUID; i++)
        {
            for (j = 0; j < 4 * NFLUID; j++)
            {
                Lvec->LF0[i][j] +=     Lvec->LF1[i][j] +     Lvec->LF2[i][j] +     Lvec->LF3[i][j] + Lvec->LF4[i][j];
                Lvec->LF1[i][j] += 2.0*Lvec->LF2[i][j] + 3.0*Lvec->LF3[i][j] + 4.0*Lvec->LF4[i][j];
                Lvec->LF2[i][j] += 3.0*Lvec->LF3[i][j] + 6.0*Lvec->LF4[i][j];
                Lvec->LF3[i][j] += 4.0*Lvec->LF4[i][j];
            }
        }
    }

    // move the wall lattice sites
    for (i = 0; i < NWALL; i++)
    {
        phase->ux0[i] += phase->wallvel * dt;
        phase->dx0[i] -= phase->wallvel * dt;
    }

    // periodic boundary conditions in x
    for (i = 0; i < NPART; i++)
    {
        pbcs(&phase->x0[i]);
    }
    for (i = 0; i < NWALL; i++)
    {
        pbcs(&phase->ux0[i]);
        pbcs(&phase->dx0[i]);
    }

    // forces, thermostat multipliers and (if needed) Jacobian at the predicted state
    force(lyap, phase, interact, Lvec, t, check);

    // ---------------------------------------------------- corrector: upper wall
    for (i = 0; i < NWALL; i++)
    {
        xcor = phase->x1[i] - phase->px0[i] * dt;
        ycor = phase->y1[i] - phase->py0[i] * dt;

        // thermostat force alpha*(p - p_ref)
        thermx = interact->alphau * (phase->px0[i] - phase->pmxu * (1 - const_velocity) - phase->wallvel * const_velocity);
        thermy = interact->alphau * (phase->py0[i] - phase->pmyu * (1 - const_velocity));

        pxcor = phase->px1[i] - (interact->fx[i] - thermx) * dt;
        pycor = phase->py1[i] - (interact->fy[i] - thermy) * dt;

        phase->x0[i] -= xcor * f01;
        phase->y0[i] -= ycor * f01;

        phase->x1[i] -= xcor;
        phase->y1[i] -= ycor;

        phase->x2[i] -= xcor * f21;
        phase->y2[i] -= ycor * f21;

        phase->x3[i] -= xcor * f31;
        phase->y3[i] -= ycor * f31;

        phase->x4[i] -= xcor * f41;
        phase->y4[i] -= ycor * f41;

        phase->px0[i] -= pxcor * f01;
        phase->py0[i] -= pycor * f01;

        phase->px1[i] -= pxcor;
        phase->py1[i] -= pycor;

        phase->px2[i] -= pxcor * f21;
        phase->py2[i] -= pycor * f21;

        phase->px3[i] -= pxcor * f31;
        phase->py3[i] -= pycor * f31;

        phase->px4[i] -= pxcor * f41;
        phase->py4[i] -= pycor * f41;
    }

    // ---------------------------------------------------- corrector: lower wall
    for (i = NWALL; i < 2 * NWALL; i++)
    {
        xcor = phase->x1[i] - phase->px0[i] * dt;
        ycor = phase->y1[i] - phase->py0[i] * dt;

        thermx = interact->alphad * (phase->px0[i] - phase->pmxd * (1 - const_velocity) + phase->wallvel * const_velocity);
        thermy = interact->alphad * (phase->py0[i] - phase->pmyd * (1 - const_velocity));

        pxcor = phase->px1[i] - (interact->fx[i] - thermx) * dt;
        pycor = phase->py1[i] - (interact->fy[i] - thermy) * dt;

        phase->x0[i] -= xcor * f01;
        phase->y0[i] -= ycor * f01;

        phase->x1[i] -= xcor;
        phase->y1[i] -= ycor;

        phase->x2[i] -= xcor * f21;
        phase->y2[i] -= ycor * f21;

        phase->x3[i] -= xcor * f31;
        phase->y3[i] -= ycor * f31;

        phase->x4[i] -= xcor * f41;
        phase->y4[i] -= ycor * f41;

        phase->px0[i] -= pxcor * f01;
        phase->py0[i] -= pycor * f01;

        phase->px1[i] -= pxcor;
        phase->py1[i] -= pycor;

        phase->px2[i] -= pxcor * f21;
        phase->py2[i] -= pycor * f21;

        phase->px3[i] -= pxcor * f31;
        phase->py3[i] -= pycor * f31;

        phase->px4[i] -= pxcor * f41;
        phase->py4[i] -= pycor * f41;
    }

    // ---------------------------------------- corrector: fluid (no thermostat)
    for (i = 2 * NWALL; i < NPART; i++)
    {
        xcor = phase->x1[i] - phase->px0[i] * dt;
        ycor = phase->y1[i] - phase->py0[i] * dt;

        pxcor = phase->px1[i] - (interact->fx[i]) * dt;
        pycor = phase->py1[i] - (interact->fy[i]) * dt;

        phase->x0[i] -= xcor * f01;
        phase->y0[i] -= ycor * f01;

        phase->x1[i] -= xcor;
        phase->y1[i] -= ycor;

        phase->x2[i] -= xcor * f21;
        phase->y2[i] -= ycor * f21;

        phase->x3[i] -= xcor * f31;
        phase->y3[i] -= ycor * f31;

        phase->x4[i] -= xcor * f41;
        phase->y4[i] -= ycor * f41;

        phase->px0[i] -= pxcor * f01;
        phase->py0[i] -= pycor * f01;

        phase->px1[i] -= pxcor;
        phase->py1[i] -= pycor;

        phase->px2[i] -= pxcor * f21;
        phase->py2[i] -= pycor * f21;

        phase->px3[i] -= pxcor * f31;
        phase->py3[i] -= pycor * f31;

        phase->px4[i] -= pxcor * f41;
        phase->py4[i] -= pycor * f41;
    }

    // ------------------------------------------- corrector: tangent vectors
    if (LYAP == 1 && t >= WARMUP)
    {
        double Lcor;

        for (i = 0; i < 4 * NPART; i++)
        {
            for (j = 0; j < 4 * NPART; j++)
            {
                Lcor = Lvec->L1[i][j] - interact->TT[i][j] * dt;

                Lvec->L0[i][j] -= Lcor * f01;
                Lvec->L1[i][j] -= Lcor;
                Lvec->L2[i][j] -= Lcor * f21;
                Lvec->L3[i][j] -= Lcor * f31;
                Lvec->L4[i][j] -= Lcor * f41;
            }
        }

        for (i = 0; i < 4 * NFLUID; i++)
        {
            for (j = 0; j < 4 * NFLUID; j++)
            {
                Lcor = Lvec->LF1[i][j] - interact->TTF[i][j] * dt;

                Lvec->LF0[i][j] -= Lcor * f01;
                Lvec->LF1[i][j] -= Lcor;
                Lvec->LF2[i][j] -= Lcor * f21;
                Lvec->LF3[i][j] -= Lcor * f31;
                Lvec->LF4[i][j] -= Lcor * f41;
            }
        }
    }

    // periodic boundary conditions in x
    for (i = 0; i < NPART; i++)
    {
        pbcs(&phase->x0[i]);
    }
    for (i = 0; i < NWALL; i++)
    {
        pbcs(&phase->ux0[i]);
        pbcs(&phase->dx0[i]);
    }
}

// Wrap an x coordinate (or an x separation) into [-Lx/2, Lx/2)
void pbcs(double *x)
{
    if (*x <= -Lx2)
    {
        *x += Lx;
    }
    else if (*x >= Lx2)
    {
        *x -= Lx;
    }
}


/* ============================================================================
 *  Output of the run parameters
 * ========================================================================== */

// Write the run parameters to Info_simulation.txt
void savefiles(Phase *phase)
{
    ofstream input2("Info_simulation.txt");

    input2 << "Fluid particles = " << NPART << endl;
    input2 << "Wall particles (total) = " << 2 * NWALL << endl;
    input2 << "System size (x,y) Lx= " << Lx << ", Ly=" << Ly << endl;
    input2 << "System density = " << rho << endl;
    input2 << "System temperature = " << phase->T << endl;
    input2 << "Particles potential = " << "WCA" << endl;
    input2 << "shear = " << phase->gammadot << endl;
    input2 << "spring constant = " << K << endl;
    input2 << "Feedback = " << Fback << endl;
    input2 << "timestep = " << dt << endl;

    input2.close();
}


/* ============================================================================
 *  Forces and Jacobian
 * ========================================================================== */

// WCA forces between all pairs that interact: upper wall-upper wall,
// lower wall-lower wall, fluid-fluid, upper wall-fluid, lower wall-fluid.
// The two walls do not interact with each other. Only x uses the minimum image.
//
// With LYAP = 1 and t >= WARMUP it also builds the Hessian of the potential
// (dudxx, dudxy, dudyy) and adds it to the dp/dq block of the Jacobian flyap.
void wcaforce(Lyapforce *lyap, Phase *phase, Interact *interact, long long int t)
{
    int i, j;
    double rx, ry, rsi, r6, rsq;
    double F, DF, partsum;

    if (t >= WARMUP && LYAP == 1)
    {
        // ---------------------------------------------- forces + Hessian
        // For a pair: F = 24 eps (sigma^2/r^2)^3 (2 (sigma^2/r^2)^3 - 1) / r^2  (force / r)
        // dudxx[i][j] = dF_ix/dx_j, and the diagonal blocks are minus the sum
        // of the off-diagonal ones.

        // upper wall - upper wall
        for (i = 0; i < NWALL - 1; i++)
        {
            for (j = i + 1; j < NWALL; j++)
            {
                rx = phase->x0[i] - phase->x0[j];
                ry = phase->y0[i] - phase->y0[j];

                pbcs(&rx);

                rsq = rx * rx + ry * ry;
                if (rsq < R2W)
                {
                    rsi = sigmaW * sigmaW / rsq;
                    r6  = rsi * rsi * rsi;
                    rsi = 1. / rsq;

                    F = 24. * epsW * rsi * r6 * (2. * r6 - 1.);

                    interact->fx[i] += rx * F;
                    interact->fy[i] += ry * F;
                    interact->fx[j] -= rx * F;
                    interact->fy[j] -= ry * F;

                    DF = 24. * epsW * rsi * r6 * (26. * r6 - 7.);

                    partsum = rsi * (F + DF);

                    lyap->dudxx[i][j] = -F + rx * rx * partsum;
                    lyap->dudxy[i][j] =      rx * ry * partsum;
                    lyap->dudyy[i][j] = -F + ry * ry * partsum;

                    lyap->dudxx[j][i] = lyap->dudxx[i][j];
                    lyap->dudxy[j][i] = lyap->dudxy[i][j];
                    lyap->dudyy[j][i] = lyap->dudyy[i][j];

                    lyap->dudxx[i][i] -= lyap->dudxx[i][j];
                    lyap->dudxy[i][i] -= lyap->dudxy[i][j];
                    lyap->dudyy[i][i] -= lyap->dudyy[i][j];

                    lyap->dudxx[j][j] -= lyap->dudxx[i][j];
                    lyap->dudxy[j][j] -= lyap->dudxy[i][j];
                    lyap->dudyy[j][j] -= lyap->dudyy[i][j];
                }
            }
        }

        // lower wall - lower wall
        for (i = NWALL; i < 2 * NWALL - 1; i++)
        {
            for (j = i + 1; j < 2 * NWALL; j++)
            {
                rx = phase->x0[i] - phase->x0[j];
                ry = phase->y0[i] - phase->y0[j];

                pbcs(&rx);

                rsq = rx * rx + ry * ry;
                if (rsq < R2W)
                {
                    rsi = sigmaW * sigmaW / rsq;
                    r6  = rsi * rsi * rsi;
                    rsi = 1. / rsq;

                    F = 24. * epsW * rsi * r6 * (2. * r6 - 1.);

                    interact->fx[i] += rx * F;
                    interact->fy[i] += ry * F;
                    interact->fx[j] -= rx * F;
                    interact->fy[j] -= ry * F;

                    DF = 24. * epsW * rsi * r6 * (26. * r6 - 7.);

                    partsum = rsi * (F + DF);

                    lyap->dudxx[i][j] = -F + rx * rx * partsum;
                    lyap->dudxy[i][j] =      rx * ry * partsum;
                    lyap->dudyy[i][j] = -F + ry * ry * partsum;

                    lyap->dudxx[j][i] = lyap->dudxx[i][j];
                    lyap->dudxy[j][i] = lyap->dudxy[i][j];
                    lyap->dudyy[j][i] = lyap->dudyy[i][j];

                    lyap->dudxx[i][i] -= lyap->dudxx[i][j];
                    lyap->dudxy[i][i] -= lyap->dudxy[i][j];
                    lyap->dudyy[i][i] -= lyap->dudyy[i][j];

                    lyap->dudxx[j][j] -= lyap->dudxx[i][j];
                    lyap->dudxy[j][j] -= lyap->dudxy[i][j];
                    lyap->dudyy[j][j] -= lyap->dudyy[i][j];
                }
            }
        }

        // fluid - fluid
        for (i = 2 * NWALL; i < NPART - 1; i++)
        {
            for (j = i + 1; j < NPART; j++)
            {
                rx = phase->x0[i] - phase->x0[j];
                ry = phase->y0[i] - phase->y0[j];

                pbcs(&rx);

                rsq = rx * rx + ry * ry;
                if (rsq < R2F)
                {
                    rsi = sigmaF * sigmaF / rsq;
                    r6  = rsi * rsi * rsi;
                    rsi = 1. / rsq;

                    F = 24. * epsF * rsi * r6 * (2. * r6 - 1.);

                    interact->fx[i] += rx * F;
                    interact->fy[i] += ry * F;
                    interact->fx[j] -= rx * F;
                    interact->fy[j] -= ry * F;

                    DF = 24. * epsF * rsi * r6 * (26. * r6 - 7.);

                    partsum = rsi * (F + DF);

                    lyap->dudxx[i][j] = -F + rx * rx * partsum;
                    lyap->dudxy[i][j] =      rx * ry * partsum;
                    lyap->dudyy[i][j] = -F + ry * ry * partsum;

                    lyap->dudxx[j][i] = lyap->dudxx[i][j];
                    lyap->dudxy[j][i] = lyap->dudxy[i][j];
                    lyap->dudyy[j][i] = lyap->dudyy[i][j];

                    lyap->dudxx[i][i] -= lyap->dudxx[i][j];
                    lyap->dudxy[i][i] -= lyap->dudxy[i][j];
                    lyap->dudyy[i][i] -= lyap->dudyy[i][j];

                    lyap->dudxx[j][j] -= lyap->dudxx[i][j];
                    lyap->dudxy[j][j] -= lyap->dudxy[i][j];
                    lyap->dudyy[j][j] -= lyap->dudyy[i][j];
                }
            }
        }

        // upper wall - fluid
        for (i = 0; i < NWALL; i++)
        {
            for (j = 2 * NWALL; j < NPART; j++)
            {
                rx = phase->x0[i] - phase->x0[j];
                ry = phase->y0[i] - phase->y0[j];

                pbcs(&rx);

                rsq = rx * rx + ry * ry;
                if (rsq < R2FW)
                {
                    rsi = sigmaFW * sigmaFW / rsq;
                    r6  = rsi * rsi * rsi;
                    rsi = 1. / rsq;

                    F = 24. * epsFW * rsi * r6 * (2. * r6 - 1.);

                    interact->fx[i] += rx * F;
                    interact->fy[i] += ry * F;
                    interact->fx[j] -= rx * F;
                    interact->fy[j] -= ry * F;

                    DF = 24. * epsFW * rsi * r6 * (26. * r6 - 7.);

                    partsum = rsi * (F + DF);

                    lyap->dudxx[i][j] = -F + rx * rx * partsum;
                    lyap->dudxy[i][j] =      rx * ry * partsum;
                    lyap->dudyy[i][j] = -F + ry * ry * partsum;

                    lyap->dudxx[j][i] = lyap->dudxx[i][j];
                    lyap->dudxy[j][i] = lyap->dudxy[i][j];
                    lyap->dudyy[j][i] = lyap->dudyy[i][j];

                    lyap->dudxx[i][i] -= lyap->dudxx[i][j];
                    lyap->dudxy[i][i] -= lyap->dudxy[i][j];
                    lyap->dudyy[i][i] -= lyap->dudyy[i][j];

                    lyap->dudxx[j][j] -= lyap->dudxx[i][j];
                    lyap->dudxy[j][j] -= lyap->dudxy[i][j];
                    lyap->dudyy[j][j] -= lyap->dudyy[i][j];
                }
            }
        }

        // lower wall - fluid
        for (i = NWALL; i < 2 * NWALL; i++)
        {
            for (j = 2 * NWALL; j < NPART; j++)
            {
                rx = phase->x0[i] - phase->x0[j];
                ry = phase->y0[i] - phase->y0[j];

                pbcs(&rx);

                rsq = rx * rx + ry * ry;
                if (rsq < R2FW)
                {
                    rsi = sigmaFW * sigmaFW / rsq;
                    r6  = rsi * rsi * rsi;
                    rsi = 1. / rsq;

                    F = 24. * epsFW * rsi * r6 * (2. * r6 - 1.);

                    interact->fx[i] += rx * F;
                    interact->fy[i] += ry * F;
                    interact->fx[j] -= rx * F;
                    interact->fy[j] -= ry * F;

                    DF = 24. * epsFW * rsi * r6 * (26. * r6 - 7.);

                    partsum = rsi * (F + DF);

                    lyap->dudxx[i][j] = -F + rx * rx * partsum;
                    lyap->dudxy[i][j] =      rx * ry * partsum;
                    lyap->dudyy[i][j] = -F + ry * ry * partsum;

                    lyap->dudxx[j][i] = lyap->dudxx[i][j];
                    lyap->dudxy[j][i] = lyap->dudxy[i][j];
                    lyap->dudyy[j][i] = lyap->dudyy[i][j];

                    lyap->dudxx[i][i] -= lyap->dudxx[i][j];
                    lyap->dudxy[i][i] -= lyap->dudxy[i][j];
                    lyap->dudyy[i][i] -= lyap->dudyy[i][j];

                    lyap->dudxx[j][j] -= lyap->dudxx[i][j];
                    lyap->dudxy[j][j] -= lyap->dudxy[i][j];
                    lyap->dudyy[j][j] -= lyap->dudyy[i][j];
                }
            }
        }

        // Hessian -> dp/dq block of the Jacobian (rows px, py; columns x, y)
        for (i = 0; i < NPART; i++)
        {
            for (j = 0; j < NPART; j++)
            {
                interact->flyap[4 * i + 2][4 * j]     += lyap->dudxx[i][j];
                interact->flyap[4 * i + 2][4 * j + 1] += lyap->dudxy[i][j];
                interact->flyap[4 * i + 3][4 * j]     += lyap->dudxy[i][j];
                interact->flyap[4 * i + 3][4 * j + 1] += lyap->dudyy[i][j];
            }
        }
    }
    else
    {
        // ---------------------------------------------- forces only

        // upper wall - upper wall
        for (i = 0; i < NWALL - 1; i++)
        {
            for (j = i + 1; j < NWALL; j++)
            {
                rx = phase->x0[i] - phase->x0[j];
                ry = phase->y0[i] - phase->y0[j];

                pbcs(&rx);

                rsq = rx * rx + ry * ry;
                if (rsq < R2W)
                {
                    rsi = sigmaW * sigmaW / rsq;
                    r6  = rsi * rsi * rsi;
                    rsi = 1. / rsq;

                    F = 24. * epsW * rsi * r6 * (2. * r6 - 1.);

                    interact->fx[i] += rx * F;
                    interact->fy[i] += ry * F;
                    interact->fx[j] -= rx * F;
                    interact->fy[j] -= ry * F;
                }
            }
        }

        // lower wall - lower wall
        for (i = NWALL; i < 2 * NWALL - 1; i++)
        {
            for (j = i + 1; j < 2 * NWALL; j++)
            {
                rx = phase->x0[i] - phase->x0[j];
                ry = phase->y0[i] - phase->y0[j];

                pbcs(&rx);

                rsq = rx * rx + ry * ry;
                if (rsq < R2W)
                {
                    rsi = sigmaW * sigmaW / rsq;
                    r6  = rsi * rsi * rsi;
                    rsi = 1. / rsq;

                    F = 24. * epsW * rsi * r6 * (2. * r6 - 1.);

                    interact->fx[i] += rx * F;
                    interact->fy[i] += ry * F;
                    interact->fx[j] -= rx * F;
                    interact->fy[j] -= ry * F;
                }
            }
        }

        // fluid - fluid
        for (i = 2 * NWALL; i < NPART - 1; i++)
        {
            for (j = i + 1; j < NPART; j++)
            {
                rx = phase->x0[i] - phase->x0[j];
                ry = phase->y0[i] - phase->y0[j];

                pbcs(&rx);

                rsq = rx * rx + ry * ry;
                if (rsq < R2F)
                {
                    rsi = sigmaF * sigmaF / rsq;
                    r6  = rsi * rsi * rsi;
                    rsi = 1. / rsq;

                    F = 24. * epsF * rsi * r6 * (2. * r6 - 1.);

                    interact->fx[i] += rx * F;
                    interact->fy[i] += ry * F;
                    interact->fx[j] -= rx * F;
                    interact->fy[j] -= ry * F;
                }
            }
        }

        // upper wall - fluid
        for (i = 0; i < NWALL; i++)
        {
            for (j = 2 * NWALL; j < NPART; j++)
            {
                rx = phase->x0[i] - phase->x0[j];
                ry = phase->y0[i] - phase->y0[j];

                pbcs(&rx);

                rsq = rx * rx + ry * ry;
                if (rsq < R2FW)
                {
                    rsi = sigmaFW * sigmaFW / rsq;
                    r6  = rsi * rsi * rsi;
                    rsi = 1. / rsq;

                    F = 24. * epsFW * rsi * r6 * (2. * r6 - 1.);

                    interact->fx[i] += rx * F;
                    interact->fy[i] += ry * F;
                    interact->fx[j] -= rx * F;
                    interact->fy[j] -= ry * F;
                }
            }
        }

        // lower wall - fluid
        for (i = NWALL; i < 2 * NWALL; i++)
        {
            for (j = 2 * NWALL; j < NPART; j++)
            {
                rx = phase->x0[i] - phase->x0[j];
                ry = phase->y0[i] - phase->y0[j];

                pbcs(&rx);

                rsq = rx * rx + ry * ry;
                if (rsq < R2FW)
                {
                    rsi = sigmaFW * sigmaFW / rsq;
                    r6  = rsi * rsi * rsi;
                    rsi = 1. / rsq;

                    F = 24. * epsFW * rsi * r6 * (2. * r6 - 1.);

                    interact->fx[i] += rx * F;
                    interact->fy[i] += ry * F;
                    interact->fx[j] -= rx * F;
                    interact->fy[j] -= ry * F;
                }
            }
        }
    }
}

// Gaussian thermostat multipliers of the two walls:
//   alpha = sum(F.p') / sum(p'.p')     with p' = p - p_ref, F' = F - <F> (const_velocity = 0)
// plus a proportional feedback term on the wall kinetic energy.
//
// With LYAP = 1 and t >= WARMUP it also adds the thermostat contribution to
// the Jacobian flyap:
//   - the diagonal (and, for const_velocity = 0, the mean-momentum) part -alpha*(dp'/dp)
//   - the terms -(p - p_ref) * d(alpha)/dq and -(p - p_ref) * d(alpha)/dp
// The derivatives of alpha w.r.t. positions come from the wall rows of flyap,
// so wcaforce() and springforce() must be called before this function.
void thermostatforce(Lyapforce *lyap, Phase *phase, Interact *interact, long long int t, ofstream &check)
{
    double px, py, fx, fy;
    double Fxx, Fxy, Fyx, Fyy;

    double adenu = 0;   // denominator (sum of p'^2), upper wall
    double anumu = 0;   // numerator   (sum of F.p'), upper wall
    double adend = 0;   // lower wall
    double anumd = 0;

    double dadendpx, danumdpx, dadendpy, danumdpy;

    int i, j, ii, jj, step;

    interact->alphau = 0;
    interact->alphad = 0;

    phase->pmxu = 0;
    phase->pmyu = 0;
    interact->fmxu = 0;
    interact->fmyu = 0;

    phase->pmxd = 0;
    phase->pmyd = 0;
    interact->fmxd = 0;
    interact->fmyd = 0;

    // ----------------------------------------------------- multipliers
    if (const_velocity == 1)
    {
        // reference momentum = imposed wall velocity
        for (i = 0; i < NWALL; i++)
        {
            px = phase->px0[i] - phase->wallvel;
            py = phase->py0[i] - 0;
            fx = interact->fx[i] - 0;
            fy = interact->fy[i] - 0;

            adenu += px * px + py * py;
            anumu += fx * px + fy * py;
        }

        interact->alphau = anumu / adenu;

        for (i = NWALL; i < 2 * NWALL; i++)
        {
            px = phase->px0[i] + phase->wallvel;
            py = phase->py0[i] - 0;
            fx = interact->fx[i] - 0;
            fy = interact->fy[i] - 0;

            adend += px * px + py * py;
            anumd += fx * px + fy * py;
        }

        interact->alphad = anumd / adend;
    }
    else
    {
        // reference momentum = instantaneous mean momentum of the wall
        // upper wall: mean momentum and mean force
        for (i = 0; i < NWALL; i++)
        {
            phase->pmxu += phase->px0[i];
            phase->pmyu += phase->py0[i];
            interact->fmxu += interact->fx[i];
            interact->fmyu += interact->fy[i];
        }

        phase->pmxu /= NWALL;
        phase->pmyu /= NWALL;
        interact->fmxu /= NWALL;
        interact->fmyu /= NWALL;

        for (i = 0; i < NWALL; i++)
        {
            px = phase->px0[i] - phase->pmxu;
            py = phase->py0[i] - phase->pmyu;
            fx = interact->fx[i] - interact->fmxu;
            fy = interact->fy[i] - interact->fmyu;

            adenu += px * px + py * py;
            anumu += fx * px + fy * py;
        }

        interact->alphau = anumu / adenu;

        // lower wall: mean momentum and mean force
        for (i = NWALL; i < 2 * NWALL; i++)
        {
            phase->pmxd += phase->px0[i];
            phase->pmyd += phase->py0[i];
            interact->fmxd += interact->fx[i];
            interact->fmyd += interact->fy[i];
        }

        phase->pmxd /= NWALL;
        phase->pmyd /= NWALL;
        interact->fmxd /= NWALL;
        interact->fmyd /= NWALL;

        for (i = NWALL; i < 2 * NWALL; i++)
        {
            px = phase->px0[i] - phase->pmxd;
            py = phase->py0[i] - phase->pmyd;
            fx = interact->fx[i] - interact->fmxd;
            fy = interact->fy[i] - interact->fmyd;

            adend += px * px + py * py;
            anumd += fx * px + fy * py;
        }

        interact->alphad = anumd / adend;
    }

    // ----------------------------------------- thermostat part of the Jacobian
    // Index mapping used below for the "walls + fluid" arrays of size NWALL+NFLUID:
    //   upper wall: i in [0, NWALL)              -> particle i            (ii = i)
    //   rest:       i in [NWALL, NWALL+NFLUID)   -> particle i + NWALL    (ii = step*NWALL + i, step = i/NWALL)
    if (t >= WARMUP && LYAP == 1)
    {
        if (const_velocity == 1)
        {
            // d(alpha_up)/d(p) of the upper-wall momenta
            for (i = 0; i < NWALL; i++)
            {
                danumdpx = interact->fx[i];
                dadendpx = 2 * (phase->px0[i] - phase->wallvel);

                danumdpy = interact->fy[i];
                dadendpy = 2 * (phase->py0[i]);

                lyap->dalphaupdpx[i] = (danumdpx * adenu - anumu * dadendpx) / (adenu * adenu);
                lyap->dalphaupdpy[i] = (danumdpy * adenu - anumu * dadendpy) / (adenu * adenu);
            }

            // d(alpha_up)/d(q) of upper-wall and fluid positions
            for (i = 0; i < NWALL + NFLUID; i++)
            {
                step = i / NWALL;
                ii = step * NWALL + i;

                for (j = 0; j < NWALL; j++)
                {
                    lyap->dalphaupdx[i] += (interact->flyap[4 * j + 2][4 * ii])     * (phase->px0[j] - phase->wallvel) + (interact->flyap[4 * j + 3][4 * ii])     * (phase->py0[j]);
                    lyap->dalphaupdy[i] += (interact->flyap[4 * j + 2][4 * ii + 1]) * (phase->px0[j] - phase->wallvel) + (interact->flyap[4 * j + 3][4 * ii + 1]) * (phase->py0[j]);
                }

                lyap->dalphaupdx[i] /= adenu;
                lyap->dalphaupdy[i] /= adenu;
            }

            // d(alpha_down)/d(p) of the lower-wall momenta
            for (i = NWALL; i < 2 * NWALL; i++)
            {
                danumdpx = interact->fx[i];
                dadendpx = 2 * (phase->px0[i] + phase->wallvel);

                danumdpy = interact->fy[i];
                dadendpy = 2 * (phase->py0[i]);

                lyap->dalphadowndpx[i - NWALL] = (danumdpx * adend - anumd * dadendpx) / (adend * adend);
                lyap->dalphadowndpy[i - NWALL] = (danumdpy * adend - anumd * dadendpy) / (adend * adend);
            }

            // d(alpha_down)/d(q) of lower-wall and fluid positions
            for (i = NWALL; i < NPART; i++)
            {
                ii = i;

                for (j = NWALL; j < 2 * NWALL; j++)
                {
                    lyap->dalphadowndx[i - NWALL] += (interact->flyap[4 * j + 2][4 * ii])     * (phase->px0[j] + phase->wallvel) + (interact->flyap[4 * j + 3][4 * ii])     * (phase->py0[j]);
                    lyap->dalphadowndy[i - NWALL] += (interact->flyap[4 * j + 2][4 * ii + 1]) * (phase->px0[j] + phase->wallvel) + (interact->flyap[4 * j + 3][4 * ii + 1]) * (phase->py0[j]);
                }

                lyap->dalphadowndx[i - NWALL] /= adend;
                lyap->dalphadowndy[i - NWALL] /= adend;
            }

            // -alpha on the momentum diagonal
            for (i = 0; i < NWALL; i++)
            {
                interact->flyap[4 * i + 2][4 * i + 2] += -interact->alphau;
                interact->flyap[4 * i + 3][4 * i + 3] += -interact->alphau;
            }

            for (i = NWALL; i < 2 * NWALL; i++)
            {
                interact->flyap[4 * i + 2][4 * i + 2] += -interact->alphad;
                interact->flyap[4 * i + 3][4 * i + 3] += -interact->alphad;
            }

            // -(p - p_ref) * d(alpha)/dq
            for (i = 0; i < NWALL; i++)
            {
                for (j = 0; j < NWALL + NFLUID; j++)
                {
                    step = j / NWALL;
                    jj = step * NWALL + j;

                    interact->flyap[4 * i + 2][4 * jj]     += -lyap->dalphaupdx[j] * (phase->px0[i] - phase->wallvel);
                    interact->flyap[4 * i + 2][4 * jj + 1] += -lyap->dalphaupdy[j] * (phase->px0[i] - phase->wallvel);

                    interact->flyap[4 * i + 3][4 * jj]     += -lyap->dalphaupdx[j] * (phase->py0[i]);
                    interact->flyap[4 * i + 3][4 * jj + 1] += -lyap->dalphaupdy[j] * (phase->py0[i]);
                }
            }

            for (i = NWALL; i < 2 * NWALL; i++)
            {
                for (j = NWALL; j < NPART; j++)
                {
                    interact->flyap[4 * i + 2][4 * j]     += -lyap->dalphadowndx[j - NWALL] * (phase->px0[i] + phase->wallvel);
                    interact->flyap[4 * i + 2][4 * j + 1] += -lyap->dalphadowndy[j - NWALL] * (phase->px0[i] + phase->wallvel);

                    interact->flyap[4 * i + 3][4 * j]     += -lyap->dalphadowndx[j - NWALL] * (phase->py0[i]);
                    interact->flyap[4 * i + 3][4 * j + 1] += -lyap->dalphadowndy[j - NWALL] * (phase->py0[i]);
                }
            }

            // -(p - p_ref) * d(alpha)/dp
            for (i = 0; i < NWALL; i++)
            {
                for (j = 0; j < NWALL; j++)
                {
                    interact->flyap[4 * i + 2][4 * j + 2] += -lyap->dalphaupdpx[j] * (phase->px0[i] - phase->wallvel);
                    interact->flyap[4 * i + 2][4 * j + 3] += -lyap->dalphaupdpy[j] * (phase->px0[i] - phase->wallvel);

                    interact->flyap[4 * i + 3][4 * j + 2] += -lyap->dalphaupdpx[j] * (phase->py0[i]);
                    interact->flyap[4 * i + 3][4 * j + 3] += -lyap->dalphaupdpy[j] * (phase->py0[i]);
                }
            }

            for (i = NWALL; i < 2 * NWALL; i++)
            {
                for (j = NWALL; j < 2 * NWALL; j++)
                {
                    interact->flyap[4 * i + 2][4 * j + 2] += -lyap->dalphadowndpx[j - NWALL] * (phase->px0[i] + phase->wallvel);
                    interact->flyap[4 * i + 2][4 * j + 3] += -lyap->dalphadowndpy[j - NWALL] * (phase->px0[i] + phase->wallvel);

                    interact->flyap[4 * i + 3][4 * j + 2] += -lyap->dalphadowndpx[j - NWALL] * (phase->py0[i]);
                    interact->flyap[4 * i + 3][4 * j + 3] += -lyap->dalphadowndpy[j - NWALL] * (phase->py0[i]);
                }
            }
        }
        else
        {
            // d(alpha_up)/d(p) of the upper-wall momenta (peculiar momenta w.r.t. the wall mean)
            for (i = 0; i < NWALL; i++)
            {
                danumdpx = 0;
                dadendpx = 0;

                danumdpy = 0;
                dadendpy = 0;

                // contribution through the mean (sums of peculiar quantities, ~0)
                for (j = 0; j < NWALL; j++)
                {
                    danumdpx += -(interact->fx[j] - interact->fmxu);
                    dadendpx += -(phase->px0[j] - phase->pmxu);

                    danumdpy += -(interact->fy[j] - interact->fmyu);
                    dadendpy += -(phase->py0[j] - phase->pmyu);
                }
                danumdpx /= NWALL;
                dadendpx /= NWALL;

                danumdpy /= NWALL;
                dadendpy /= NWALL;

                // direct contribution
                danumdpx += interact->fx[i] - interact->fmxu;
                dadendpx += (phase->px0[i] - phase->pmxu);
                dadendpx *= 2;

                danumdpy += interact->fy[i] - interact->fmyu;
                dadendpy += (phase->py0[i] - phase->pmyu);
                dadendpy *= 2;

                lyap->dalphaupdpx[i] = (danumdpx * adenu - anumu * dadendpx) / (adenu * adenu);
                lyap->dalphaupdpy[i] = (danumdpy * adenu - anumu * dadendpy) / (adenu * adenu);
            }

            // d(alpha_up)/d(q) of upper-wall and fluid positions, using F - <F>
            for (i = 0; i < NWALL + NFLUID; i++)
            {
                step = i / NWALL;
                ii = step * NWALL + i;

                // mean over the wall of dF/dq
                Fxx = 0;
                Fxy = 0;
                Fyx = 0;
                Fyy = 0;

                for (j = 0; j < NWALL; j++)
                {
                    Fxx += interact->flyap[4 * j + 2][4 * ii];
                    Fxy += interact->flyap[4 * j + 2][4 * ii + 1];
                    Fyx += interact->flyap[4 * j + 3][4 * ii];
                    Fyy += interact->flyap[4 * j + 3][4 * ii + 1];
                }

                Fxx /= NWALL;
                Fxy /= NWALL;
                Fyx /= NWALL;
                Fyy /= NWALL;

                for (j = 0; j < NWALL; j++)
                {
                    lyap->dalphaupdx[i] += (interact->flyap[4 * j + 2][4 * ii]     - Fxx) * (phase->px0[j] - phase->pmxu) + (interact->flyap[4 * j + 3][4 * ii]     - Fyx) * (phase->py0[j] - phase->pmyu);
                    lyap->dalphaupdy[i] += (interact->flyap[4 * j + 2][4 * ii + 1] - Fxy) * (phase->px0[j] - phase->pmxu) + (interact->flyap[4 * j + 3][4 * ii + 1] - Fyy) * (phase->py0[j] - phase->pmyu);
                }

                lyap->dalphaupdx[i] /= adenu;
                lyap->dalphaupdy[i] /= adenu;
            }

            // d(alpha_down)/d(p) of the lower-wall momenta
            for (i = NWALL; i < 2 * NWALL; i++)
            {
                danumdpx = 0;
                dadendpx = 0;

                danumdpy = 0;
                dadendpy = 0;

                for (j = NWALL; j < 2 * NWALL; j++)
                {
                    danumdpx += -(interact->fx[j] - interact->fmxd);
                    dadendpx += -(phase->px0[j] - phase->pmxd);

                    danumdpy += -(interact->fy[j] - interact->fmyd);
                    dadendpy += -(phase->py0[j] - phase->pmyd);
                }
                danumdpx /= NWALL;
                dadendpx /= NWALL;

                danumdpy /= NWALL;
                dadendpy /= NWALL;

                danumdpx += interact->fx[i] - interact->fmxd;
                dadendpx += (phase->px0[i] - phase->pmxd);
                dadendpx *= 2;

                danumdpy += interact->fy[i] - interact->fmyd;
                dadendpy += (phase->py0[i] - phase->pmyd);
                dadendpy *= 2;

                lyap->dalphadowndpx[i - NWALL] = (danumdpx * adend - anumd * dadendpx) / (adend * adend);
                lyap->dalphadowndpy[i - NWALL] = (danumdpy * adend - anumd * dadendpy) / (adend * adend);
            }

            // d(alpha_down)/d(q) of lower-wall and fluid positions
            for (i = NWALL; i < NPART; i++)
            {
                ii = i;

                Fxx = 0;
                Fxy = 0;
                Fyx = 0;
                Fyy = 0;

                for (j = NWALL; j < 2 * NWALL; j++)
                {
                    Fxx += interact->flyap[4 * j + 2][4 * ii];
                    Fxy += interact->flyap[4 * j + 2][4 * ii + 1];
                    Fyx += interact->flyap[4 * j + 3][4 * ii];
                    Fyy += interact->flyap[4 * j + 3][4 * ii + 1];
                }

                Fxx /= NWALL;
                Fxy /= NWALL;
                Fyx /= NWALL;
                Fyy /= NWALL;

                for (j = NWALL; j < 2 * NWALL; j++)
                {
                    lyap->dalphadowndx[i - NWALL] += (interact->flyap[4 * j + 2][4 * ii]     - Fxx) * (phase->px0[j] - phase->pmxd) + (interact->flyap[4 * j + 3][4 * ii]     - Fyx) * (phase->py0[j] - phase->pmyd);
                    lyap->dalphadowndy[i - NWALL] += (interact->flyap[4 * j + 2][4 * ii + 1] - Fxy) * (phase->px0[j] - phase->pmxd) + (interact->flyap[4 * j + 3][4 * ii + 1] - Fyy) * (phase->py0[j] - phase->pmyd);
                }

                lyap->dalphadowndx[i - NWALL] /= adend;
                lyap->dalphadowndy[i - NWALL] /= adend;
            }

            // -alpha * d(p - <p>)/dp = -alpha * (delta_ij - 1/NWALL)
            for (i = 0; i < NWALL; i++)
            {
                for (j = 0; j < NWALL; j++)
                {
                    interact->flyap[4 * i + 2][4 * j + 2] += interact->alphau / NWALL;
                    interact->flyap[4 * i + 3][4 * j + 3] += interact->alphau / NWALL;
                }
                interact->flyap[4 * i + 2][4 * i + 2] += -interact->alphau;
                interact->flyap[4 * i + 3][4 * i + 3] += -interact->alphau;
            }

            for (i = NWALL; i < 2 * NWALL; i++)
            {
                for (j = NWALL; j < 2 * NWALL; j++)
                {
                    interact->flyap[4 * i + 2][4 * j + 2] += interact->alphad / NWALL;
                    interact->flyap[4 * i + 3][4 * j + 3] += interact->alphad / NWALL;
                }
                interact->flyap[4 * i + 2][4 * i + 2] += -interact->alphad;
                interact->flyap[4 * i + 3][4 * i + 3] += -interact->alphad;
            }

            // -(p - <p>) * d(alpha)/dq
            for (i = 0; i < NWALL; i++)
            {
                for (j = 0; j < NWALL + NFLUID; j++)
                {
                    step = j / NWALL;
                    jj = step * NWALL + j;

                    interact->flyap[4 * i + 2][4 * jj]     += -lyap->dalphaupdx[j] * (phase->px0[i] - phase->pmxu);
                    interact->flyap[4 * i + 2][4 * jj + 1] += -lyap->dalphaupdy[j] * (phase->px0[i] - phase->pmxu);

                    interact->flyap[4 * i + 3][4 * jj]     += -lyap->dalphaupdx[j] * (phase->py0[i] - phase->pmyu);
                    interact->flyap[4 * i + 3][4 * jj + 1] += -lyap->dalphaupdy[j] * (phase->py0[i] - phase->pmyu);
                }
            }

            for (i = NWALL; i < 2 * NWALL; i++)
            {
                for (j = NWALL; j < NPART; j++)
                {
                    interact->flyap[4 * i + 2][4 * j]     += -lyap->dalphadowndx[j - NWALL] * (phase->px0[i] - phase->pmxd);
                    interact->flyap[4 * i + 2][4 * j + 1] += -lyap->dalphadowndy[j - NWALL] * (phase->px0[i] - phase->pmxd);

                    interact->flyap[4 * i + 3][4 * j]     += -lyap->dalphadowndx[j - NWALL] * (phase->py0[i] - phase->pmyd);
                    interact->flyap[4 * i + 3][4 * j + 1] += -lyap->dalphadowndy[j - NWALL] * (phase->py0[i] - phase->pmyd);
                }
            }

            // -(p - <p>) * d(alpha)/dp
            for (i = 0; i < NWALL; i++)
            {
                for (j = 0; j < NWALL; j++)
                {
                    interact->flyap[4 * i + 2][4 * j + 2] += -lyap->dalphaupdpx[j] * (phase->px0[i] - phase->pmxu);
                    interact->flyap[4 * i + 2][4 * j + 3] += -lyap->dalphaupdpy[j] * (phase->px0[i] - phase->pmxu);

                    interact->flyap[4 * i + 3][4 * j + 2] += -lyap->dalphaupdpx[j] * (phase->py0[i] - phase->pmyu);
                    interact->flyap[4 * i + 3][4 * j + 3] += -lyap->dalphaupdpy[j] * (phase->py0[i] - phase->pmyu);
                }
            }

            for (i = NWALL; i < 2 * NWALL; i++)
            {
                for (j = NWALL; j < 2 * NWALL; j++)
                {
                    interact->flyap[4 * i + 2][4 * j + 2] += -lyap->dalphadowndpx[j - NWALL] * (phase->px0[i] - phase->pmxd);
                    interact->flyap[4 * i + 2][4 * j + 3] += -lyap->dalphadowndpy[j - NWALL] * (phase->px0[i] - phase->pmxd);

                    interact->flyap[4 * i + 3][4 * j + 2] += -lyap->dalphadowndpx[j - NWALL] * (phase->py0[i] - phase->pmyd);
                    interact->flyap[4 * i + 3][4 * j + 3] += -lyap->dalphadowndpy[j - NWALL] * (phase->py0[i] - phase->pmyd);
                }
            }
        }
    }

    // ------------------------------------------- proportional feedback term
    interact->alphau += Fback * ((adenu - (2. * NWALL - Free_wall) * phase->T) / ((2. * NWALL - Free_wall) * phase->T));
    interact->alphad += Fback * ((adend - (2. * NWALL - Free_wall) * phase->T) / ((2. * NWALL - Free_wall) * phase->T));

    // running sums for check.txt
    if (t >= WARMUP)
    {
        interact->avalphau += interact->alphau;
        interact->avalphad += interact->alphad;
        phase->adenu += adenu;
        phase->adend += adend;
    }
}

// Harmonic springs (constant K) tethering each wall particle to its lattice
// site. With LYAP = 1 and t >= WARMUP, adds -K to the dp/dq diagonal of flyap.
void springforce(Lyapforce *lyap, Phase *phase, Interact *interact, long long int t)
{
    int i;
    double rx, ry;

    // upper wall
    for (i = 0; i < NWALL; i++)
    {
        rx = phase->x0[i] - phase->ux0[i];
        ry = phase->y0[i] - phase->uy0[i];

        pbcs(&rx);

        interact->fx[i] -= rx * K;
        interact->fy[i] -= ry * K;
    }

    // lower wall
    for (i = NWALL; i < 2 * NWALL; i++)
    {
        rx = phase->x0[i] - phase->dx0[i - NWALL];
        ry = phase->y0[i] - phase->dy0[i - NWALL];

        pbcs(&rx);

        interact->fx[i] -= rx * K;
        interact->fy[i] -= ry * K;
    }

    if (t >= WARMUP && LYAP == 1)
    {
        for (i = 0; i < NWALL; i++)
        {
            interact->flyap[4 * i + 2][4 * i]     += -K;
            interact->flyap[4 * i + 3][4 * i + 1] += -K;
        }
        for (i = NWALL; i < 2 * NWALL; i++)
        {
            interact->flyap[4 * i + 2][4 * i]     += -K;
            interact->flyap[4 * i + 3][4 * i + 1] += -K;
        }
    }
}


/* ============================================================================
 *  Lyapunov exponents
 * ========================================================================== */

// Gram-Schmidt re-orthonormalisation of the tangent vectors (columns of L0 and
// LF0). The metric weights positions by FMRAT^2 relative to momenta.
// Each vector is rescaled to norm NLYAP, and log(norm/NLYAP) is stored in
// lyapexp (flyapexp), i.e. the log expansion over the last step.
void orthonormalization(Lvec *lvec)
{
    double dot, norm, scale;
    int i, j, k;

    // ------------------------------------------------------------ full system
    for (i = 0; i < 4 * NPART; i++)
    {
        // remove the components along the previous vectors
        for (j = 0; j < i; j++)
        {
            dot = 0;
            norm = 0;

            for (k = 0; k < NPART; k++)
            {
                dot  += (lvec->L0[4*k][j]*lvec->L0[4*k][i] + lvec->L0[4*k+1][j]*lvec->L0[4*k+1][i])*FMRAT2 + lvec->L0[4*k+2][j]*lvec->L0[4*k+2][i] + lvec->L0[4*k+3][j]*lvec->L0[4*k+3][i];
                norm += (lvec->L0[4*k][j]*lvec->L0[4*k][j] + lvec->L0[4*k+1][j]*lvec->L0[4*k+1][j])*FMRAT2 + lvec->L0[4*k+2][j]*lvec->L0[4*k+2][j] + lvec->L0[4*k+3][j]*lvec->L0[4*k+3][j];
            }

            dot /= norm;

            for (k = 0; k < 4 * NPART; k++)
            {
                lvec->L0[k][i] -= dot * lvec->L0[k][j];
            }
        }

        // renormalise and record the expansion
        scale = 1;
        norm = 0;

        for (k = 0; k < NPART; k++)
        {
            norm += (lvec->L0[4*k][i]*lvec->L0[4*k][i] + lvec->L0[4*k+1][i]*lvec->L0[4*k+1][i])*FMRAT2 + lvec->L0[4*k+2][i]*lvec->L0[4*k+2][i] + lvec->L0[4*k+3][i]*lvec->L0[4*k+3][i];
        }

        if (norm > MINLYAP)
        {
            scale = NLYAP / sqrt(norm);
            lvec->lyapexp[i] = -log(scale);
        }

        for (k = 0; k < 4 * NPART; k++)
        {
            lvec->L0[k][i] *= scale;
        }
    }

    // ------------------------------------------------------------ fluid block
    for (i = 0; i < 4 * NFLUID; i++)
    {
        for (j = 0; j < i; j++)
        {
            dot = 0;
            norm = 0;

            for (k = 0; k < NFLUID; k++)
            {
                dot  += (lvec->LF0[4*k][j]*lvec->LF0[4*k][i] + lvec->LF0[4*k+1][j]*lvec->LF0[4*k+1][i])*FMRAT2 + lvec->LF0[4*k+2][j]*lvec->LF0[4*k+2][i] + lvec->LF0[4*k+3][j]*lvec->LF0[4*k+3][i];
                norm += (lvec->LF0[4*k][j]*lvec->LF0[4*k][j] + lvec->LF0[4*k+1][j]*lvec->LF0[4*k+1][j])*FMRAT2 + lvec->LF0[4*k+2][j]*lvec->LF0[4*k+2][j] + lvec->LF0[4*k+3][j]*lvec->LF0[4*k+3][j];
            }

            dot /= norm;

            for (k = 0; k < 4 * NFLUID; k++)
            {
                lvec->LF0[k][i] -= dot * lvec->LF0[k][j];
            }
        }

        scale = 1;
        norm = 0;

        for (k = 0; k < NFLUID; k++)
        {
            norm += (lvec->LF0[4*k][i]*lvec->LF0[4*k][i] + lvec->LF0[4*k+1][i]*lvec->LF0[4*k+1][i])*FMRAT2 + lvec->LF0[4*k+2][i]*lvec->LF0[4*k+2][i] + lvec->LF0[4*k+3][i]*lvec->LF0[4*k+3][i];
        }

        if (norm > MINLYAP)
        {
            scale = NLYAP / sqrt(norm);
            lvec->flyapexp[i] = -log(scale);
        }

        for (k = 0; k < 4 * NFLUID; k++)
        {
            lvec->LF0[k][i] *= scale;
        }
    }
}

// Zero the accumulators and the tangent vectors, then start from a diagonal
// basis: kickq on the position components and kickp on the momentum components.
void lyapinitialize(Lvec *lvec)
{
    int i, j;

    // full system
    for (i = 0; i < 4 * NPART; i++)
    {
        lvec->lyap2average[i] = 0;
        lvec->lyapaverage[i] = 0;
        lvec->lyapexp[i] = 0;
    }

    for (i = 0; i < 4 * NPART; i++)
    {
        for (j = 0; j < 4 * NPART; j++)
        {
            lvec->L0[i][j] = 0;
            lvec->L1[i][j] = 0;
            lvec->L2[i][j] = 0;
            lvec->L3[i][j] = 0;
            lvec->L4[i][j] = 0;
        }
    }

    // fluid block
    for (i = 0; i < 4 * NFLUID; i++)
    {
        lvec->flyap2average[i] = 0;
        lvec->flyapaverage[i] = 0;
        lvec->flyapexp[i] = 0;
    }

    for (i = 0; i < 4 * NFLUID; i++)
    {
        for (j = 0; j < 4 * NFLUID; j++)
        {
            lvec->LF0[i][j] = 0;
            lvec->LF1[i][j] = 0;
            lvec->LF2[i][j] = 0;
            lvec->LF3[i][j] = 0;
            lvec->LF4[i][j] = 0;
        }
    }

    // initial diagonal basis
    for (i = 0; i < NPART; i++)
    {
        lvec->L0[4 * i][4 * i] = kickq;
        lvec->L0[4 * i + 1][4 * i + 1] = kickq;
        lvec->L0[4 * i + 2][4 * i + 2] = kickp;
        lvec->L0[4 * i + 3][4 * i + 3] = kickp;
    }

    for (i = 0; i < NFLUID; i++)
    {
        lvec->LF0[4 * i][4 * i] = kickq;
        lvec->LF0[4 * i + 1][4 * i + 1] = kickq;
        lvec->LF0[4 * i + 2][4 * i + 2] = kickp;
        lvec->LF0[4 * i + 3][4 * i + 3] = kickp;
    }
}

// Add the last-step expansions (and their squares) to the running sums
void average(Lvec *lvec)
{
    int i;
    for (i = 0; i < 4 * NPART; i++)
    {
        lvec->lyapaverage[i] += lvec->lyapexp[i];
        lvec->lyap2average[i] += lvec->lyapexp[i] * lvec->lyapexp[i];
    }
    for (i = 0; i < 4 * NFLUID; i++)
    {
        lvec->flyapaverage[i] += lvec->flyapexp[i];
        lvec->flyap2average[i] += lvec->flyapexp[i] * lvec->flyapexp[i];
    }
}

// Sort the exponents in decreasing order of their running sum, swapping the
// corresponding tangent vectors (and their Gear derivatives) as well
void sortlyap(Lvec *lvec)
{
    int i, j, k;

    // full system
    for (i = 0; i < 4 * NPART - 1; i++)
    {
        for (j = i + 1; j < 4 * NPART; j++)
        {
            if (lvec->lyapaverage[i] < lvec->lyapaverage[j])
            {
                swap(lvec->lyapaverage[i], lvec->lyapaverage[j]);
                swap(lvec->lyap2average[i], lvec->lyap2average[j]);
                swap(lvec->lyapexp[i], lvec->lyapexp[j]);
                for (k = 0; k < 4 * NPART; k++)
                {
                    swap(lvec->L0[k][i], lvec->L0[k][j]);
                    swap(lvec->L1[k][i], lvec->L1[k][j]);
                    swap(lvec->L2[k][i], lvec->L2[k][j]);
                    swap(lvec->L3[k][i], lvec->L3[k][j]);
                    swap(lvec->L4[k][i], lvec->L4[k][j]);
                }
            }
        }
    }

    // fluid block
    for (i = 0; i < 4 * NFLUID - 1; i++)
    {
        for (j = i + 1; j < 4 * NFLUID; j++)
        {
            if (lvec->flyapaverage[i] < lvec->flyapaverage[j])
            {
                swap(lvec->flyapaverage[i], lvec->flyapaverage[j]);
                swap(lvec->flyap2average[i], lvec->flyap2average[j]);
                swap(lvec->flyapexp[i], lvec->flyapexp[j]);
                for (k = 0; k < 4 * NFLUID; k++)
                {
                    swap(lvec->LF0[k][i], lvec->LF0[k][j]);
                    swap(lvec->LF1[k][i], lvec->LF1[k][j]);
                    swap(lvec->LF2[k][i], lvec->LF2[k][j]);
                    swap(lvec->LF3[k][i], lvec->LF3[k][j]);
                    swap(lvec->LF4[k][i], lvec->LF4[k][j]);
                }
            }
        }
    }
}

// Forces on all particles and wall thermostat multipliers. With LYAP = 1 and
// t >= WARMUP also builds the Jacobian flyap and the tangent-vector time
// derivatives TT = flyap*L0 and TTF = (fluid block of flyap)*LF0.
void force(Lyapforce *lyap, Phase *phase, Interact *interact, Lvec *lvec, long long int t, ofstream &check)
{
    double dot;
    int i, j, k;

    // reset forces
    for (i = 0; i < NPART; i++)
    {
        interact->fx[i] = 0.;
        interact->fy[i] = 0.;
    }

    // reset the Jacobian and the derivative arrays
    if (t >= WARMUP && LYAP == 1)
    {
        for (i = 0; i < 4 * NPART; i++)
        {
            for (j = 0; j < 4 * NPART; j++)
            {
                interact->flyap[i][j] = 0;
            }
        }

        for (i = 0; i < NWALL; i++)
        {
            lyap->dalphaupdpx[i] = 0;
            lyap->dalphaupdpy[i] = 0;

            lyap->dalphadowndpx[i] = 0;
            lyap->dalphadowndpy[i] = 0;
        }

        for (i = 0; i < NWALL + NFLUID; i++)
        {
            lyap->dalphaupdx[i] = 0;
            lyap->dalphaupdy[i] = 0;

            lyap->dalphadowndx[i] = 0;
            lyap->dalphadowndy[i] = 0;
        }

        for (i = 0; i < NPART; i++)
        {
            for (j = 0; j < NPART; j++)
            {
                lyap->dudxx[i][j] = 0;
                lyap->dudxy[i][j] = 0;
                lyap->dudyy[i][j] = 0;
            }
        }
    }

    // the order matters: thermostatforce uses the forces and Jacobian rows built by the first two
    wcaforce(lyap, phase, interact, t);
    springforce(lyap, phase, interact, t);
    thermostatforce(lyap, phase, interact, t, check);

    if (t >= WARMUP && LYAP == 1)
    {
        // dq/dp block: dq/dt = p (unit mass)
        for (i = 0; i < NPART; i++)
        {
            interact->flyap[4 * i][4 * i + 2] += 1;
            interact->flyap[4 * i + 1][4 * i + 3] += 1;
        }

        // TT = flyap * L0
        for (i = 0; i < 4 * NPART; i++)
        {
            for (j = 0; j < 4 * NPART; j++)
            {
                dot = 0;
                for (k = 0; k < 4 * NPART; k++)
                {
                    dot += interact->flyap[i][k] * lvec->L0[k][j];
                }
                interact->TT[i][j] = dot;
            }
        }

        // TTF = (fluid block of flyap) * LF0; the fluid block starts at row/column 4*(2*NWALL)
        for (i = 0; i < 4 * NFLUID; i++)
        {
            for (j = 0; j < 4 * NFLUID; j++)
            {
                dot = 0;
                for (k = 0; k < 4 * NFLUID; k++)
                {
                    dot += interact->flyap[i + 8 * NWALL][k + 8 * NWALL] * lvec->LF0[k][j];
                }
                interact->TTF[i][j] = dot;
            }
        }
    }
}


/* ============================================================================
 *  Profiles
 * ========================================================================== */

// Accumulate px and px^2 + py^2 of every particle in bins of y.
// The bins span [-FRAC/2*Ly, FRAC/2*Ly] (nbin bins), separately for the
// fluid and the two walls.
void setprofile(Phase *phase, Profile *profile, long long int t)
{
    int j, jp;

    double bin = FRAC * Ly / nbin;

    // upper wall
    for (j = 0; j < NWALL; j++)
    {
        jp = (phase->y0[j] + FRAC / 2 * Ly) / bin;

        profile->vup[jp] += phase->px0[j];
        profile->Kinup[jp] += phase->px0[j] * phase->px0[j] + phase->py0[j] * phase->py0[j];
        profile->usample[jp]++;
    }

    // lower wall
    for (j = NWALL; j < 2 * NWALL; j++)
    {
        jp = (phase->y0[j] + FRAC / 2 * Ly) / bin;

        profile->vdown[jp] += phase->px0[j];
        profile->Kindown[jp] += phase->px0[j] * phase->px0[j] + phase->py0[j] * phase->py0[j];
        profile->dsample[jp]++;
    }

    // fluid
    for (j = 2 * NWALL; j < NPART; j++)
    {
        jp = (phase->y0[j] + FRAC / 2 * Ly) / bin;

        profile->v[jp] += phase->px0[j];
        profile->Kin[jp] += phase->px0[j] * phase->px0[j] + phase->py0[j] * phase->py0[j];
        profile->nsample[jp]++;
    }
}


/* ============================================================================
 *  Correlation integral
 * ========================================================================== */

// Grassberger-Procaccia correlation integral of the N stored fluid states.
//   1. scale every coordinate by its standard deviation
//   2. build nr radii r_i = D / base^i, where D is the diagonal of the bounding
//      box of the scaled data (the last radius is 0)
//   3. for every pair of samples, find the bin of their squared distance
//   4. cumulative sum -> C(r_i) = fraction of pairs closer than r_i
// Writes corr.txt: r  C(r)
void computecorr(double *corr)
{
    long long int i, j;

    // ------------------------------------------------ standardise the coordinates
    double stdev[dim];
    double stmean[dim];

    for (i = 0; i < dim; i++)
    {
        stdev[i] = 0;
        stmean[i] = 0;
        for (j = 0; j < N; j++)
        {
            stmean[i] += corr[dim * j + i];
            stdev[i] += corr[dim * j + i] * corr[dim * j + i];
        }

        stmean[i] /= N;
        stdev[i] /= (N - 1);
        stdev[i] += -(N / (N - 1)) * stmean[i] * stmean[i];
        stdev[i] = sqrt(stdev[i]);

        for (j = 0; j < N; j++)
        {
            corr[dim * j + i] /= stdev[i];
        }
    }

    // ---------------------------------------- bounding box of the scaled data
    double maxmin[dim][2];   // [min, max] of each coordinate

    for (i = 0; i < dim; i++)
    {
        maxmin[i][0] = corr[i];
        maxmin[i][1] = corr[i];
        for (j = 1; j < N; j++)
        {
            if (corr[dim * j + i] > maxmin[i][1])
            {
                maxmin[i][1] = corr[dim * j + i];
            }
            if (corr[dim * j + i] < maxmin[i][0])
            {
                maxmin[i][0] = corr[dim * j + i];
            }
        }
    }

    double norm = 0;   // diagonal of the bounding box

    for (i = 0; i < dim; i++)
    {
        norm += (maxmin[i][1] - maxmin[i][0]) * (maxmin[i][1] - maxmin[i][0]);
    }

    norm = sqrt(norm);

    // ---------------------------------------------------- logarithmic radii
    double base = 1.125;
    int nr = 1000;

    double r0;
    int i0;
    int k;

    double *r       = new double[nr];
    double *r2      = new double[nr];
    double *corrint = new double[nr];
    for (i = 0; i < nr - 1; i++)
    {
        r[i] = norm / pow(base, i);
        r2[i] = r[i] * r[i];
        corrint[i] = 0;
    }
    double r2max = r2[0];
    r[nr - 1] = 0;
    r2[nr - 1] = 0;
    corrint[nr - 1] = 0;

    // ------------------------------------------- histogram of pair distances
    // corrint[i0] counts the pairs with r2[i0+1] <= d^2 < r2[i0]
    for (i = 0; i < N - 1; i++)
    {
        for (j = i + 1; j < N; j++)
        {
            r0 = 0;

            for (k = 0; k < dim; k++)
            {
                r0 += (corr[dim * i + k] - corr[dim * j + k]) * (corr[dim * i + k] - corr[dim * j + k]);
            }
            if (r0 < r2max)
            {
                i0 = 0;
                while (r2[i0] > r0)
                {
                    i0++;
                }
                i0--;

                corrint[i0]++;
            }
        }
    }

    // ------------------------------------- cumulative sum and normalisation
    for (i = nr - 1; i > 0; i--)
    {
        corrint[i - 1] += corrint[i];
    }

    long long int normalization = N * (N - 1) / 2;

    for (i = 0; i < nr; i++)
    {
        corrint[i] /= normalization;
    }

    ofstream fcorr("corr.txt");

    for (i = 0; i < nr; i++)
    {
        fcorr << r[i] << " " << corrint[i] << endl;
    }

    fcorr.close();
}

// Random intervals between correlation samples:
//   3/4*FREQ + exponential with mean FREQ/4  (mean interval = FREQ)
// The fixed minimum spacing decorrelates consecutive samples; the random part
// avoids sampling in phase with any periodicity of the dynamics.
void generate_times(long long int *v)
{
    double vtemp;
    mt19937_64 rng;
    uint64_t timeSeed = chrono::high_resolution_clock::now().time_since_epoch().count();
    seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);
    uniform_real_distribution<double> unif(0, 1);
    int nSimulations = TOTSAMPLE;
    for (int i = 0; i < nSimulations; i++)
    {
        double currentRandomNumber = unif(rng);
        vtemp = (long long int)(round((-log(1 - currentRandomNumber) * (double)FREQ / 4)));
        v[i] = llrint(vtemp) + 3 * FREQ / 4;
    }
}
