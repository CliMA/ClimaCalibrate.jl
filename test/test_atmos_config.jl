import CalibrateAtmos
using Test

# Tests for ensuring CalibrateAtmos sets AtmosConfig correctly.

member_path = joinpath("test_output", "iteration_001", "member_001")
file_path = joinpath(member_path, "parameters.toml")
mkpath(dirname(file_path))
touch(file_path)

config_dict = Dict{Any, Any}(
    "restart_file" => joinpath(
        "experiments",
        "sphere_held_suarez_rhoe_equilmoist",
        "day200.0.hdf5",
    ),
    "dt_save_to_disk" => "100days",
    "moist" => "equil",
    "forcing" => "held_suarez",
    "output_dir" => "test_output",
)

atmos_config = CalibrateAtmos.get_atmos_config(1, 1, config_dict)
(; parsed_args) = atmos_config

@testset "Atmos Configuration" begin
    @test parsed_args["moist"] == "equil"
    @test parsed_args["toml"] == [file_path]
    @test parsed_args["output_dir"] == member_path
    @test ENV["RESTART_FILE"] == config_dict["restart_file"]
end

rm(file_path)
