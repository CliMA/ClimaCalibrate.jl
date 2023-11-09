# Getting Started

A good way to get started is to run the initial experiment, `sphere_held_suarez_rhoe_equilmoist`.
It is a perfect-model calibration, serving as a test case for the initial pipeline.

This experiment runs the Held-Suarez configuration, estimating the parameter `equator_pole_temperature_gradient_wet`.
By default, it runs 10 ensemble members for 3 iterations. 

To run this experiment:
1. Log onto the Caltech HPC
2. Clone CalibrateAtmos.jl and `cd` into the repository.
3. Run: `bash experiments/pipeline.sh sphere_held_suarez_rhoe_equilmoist 8`. This will run the `sphere_held_suarez_rhoe_equilmoist` experiment with 8 tasks per ensemble member.

