# Getting Started

Before you run, make sure your system meets the dependencies of [CalibrateEmulateSample.jl](https://clima.github.io/CalibrateEmulateSample.jl/dev/installation_instructions/). TODO: remove if workaround

## HPC Cluster
A good way to get started is to run the initial experiment, `sphere_held_suarez_rhoe_equilmoist`.
It is a perfect-model calibration, serving as a test case for the initial pipeline.

This experiment runs the Held-Suarez configuration, estimating the parameter `equator_pole_temperature_gradient_wet`.
By default, it runs 10 ensemble members for 3 iterations.

To run this experiment:
1. Log onto the Caltech HPC
2. Clone CalibrateAtmos.jl and `cd` into the repository.
3. Run: `bash calibrate.sh -n 10 -c 8 sphere_held_suarez_rhoe_equilmoist`. This will run the `sphere_held_suarez_rhoe_equilmoist` experiment with 10 tasks per ensemble member.

## Local Machine
To run an experiment on your local machine, you can use the `pipeline.jl` script. This is recommended for more lightweight experiments, such as the `surface_fluxes_perfect_model` experiment, which uses the [SurfaceFluxes.jl](https://github.com/CliMA/SurfaceFluxes.jl) package to generate a physical model that calculates the Monin Obukhov turbulent surface fluxes based on idealized atmospheric and surface conditions. Since this is a "perfect model" example, the same model is used to generate synthetic observations using its default parameters and a small amount of noise. These synthetic observations are considered to be the ground truth, which is used to assess the model ensembles' performance when parameters are drawn from the prior parameter distributions. To run this experiment, you can use the following command from terminal to run an interactive run:

```bash
julia -i pipeline.jl surface_fluxes_perfect_model
```

This pipeline mirrors the pipeline of the bash srcipts, and the same example can be run on the HPC cluster if needed:

```bash
bash calibrate.sh surface_fluxes_perfect_model 8
```

The experiments (such as `surface_fluxes_perfect_model`) can be equally defined within the component model repos (in this case, `SurfaceFluxes.jl`), so that the internals of `CalibrateAtmos.jl` do not explicitly depend on component models.
