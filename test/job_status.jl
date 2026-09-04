using Test
import ClimaCalibrate
const Backend = ClimaCalibrate.Backend
using ClimaCalibrate.Backend: PENDING, RUNNING, COMPLETED, FAILED

# The state parsing is pure, so it is tested here rather than only from a real
# cluster on Buildkite. A backend that cannot report a failed ensemble member
# lets a crashed calibration look successful.

@testset "Slurm job states" begin
    parse = Backend._parse_slurm_state

    @test parse("COMPLETED") == COMPLETED

    for state in (
        "FAILED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
        "REVOKED",
    )
        @test parse(state) == FAILED
    end

    # Slurm appends the cancelling user to this one
    @test parse("CANCELLED by 40826") == FAILED

    @test parse("PENDING") == PENDING
    @test parse("CONFIGURING") == PENDING
    @test parse("RUNNING") == RUNNING
    @test parse("COMPLETING") == RUNNING

    @test parse("  RUNNING\n") == RUNNING
    @test parse("running") == RUNNING

    # An unrecognized or absent state is not a success
    @test isnothing(parse(""))
    @test isnothing(parse("   "))
    @test isnothing(parse("SOME_FUTURE_STATE"))
end

@testset "PBS job states" begin
    parse = Backend._parse_pbs_state

    @test parse("job_state = R") == RUNNING
    @test parse("job_state = E") == RUNNING
    # Queued and held jobs are waiting to be scheduled, not running
    @test parse("job_state = Q") == PENDING
    @test parse("job_state = H") == PENDING
    @test parse("job_state = F") == COMPLETED

    # PBS reports a finished job as F whether or not it succeeded; substate 93
    # is the one that means it exited with an error
    @test parse("job_state = F|substate = 93") == FAILED
    @test parse("job_state = F|substate = 92") == COMPLETED
    @test parse("job_state = R|substate = 93") == RUNNING

    @test parse(
        "Job Id: 1234.desched1\n    job_state = R\n    substate = 42",
    ) == RUNNING

    @test isnothing(parse(""))
    @test isnothing(parse("Job Id: 1234.desched1"))
    @test isnothing(parse("job_state = ?"))
end

@testset "Job status predicates" begin
    @test Backend.ispending(PENDING)
    @test Backend.isrunning(RUNNING)
    @test Backend.issuccess(COMPLETED)
    @test Backend.isfailed(FAILED)

    @test !Backend.issuccess(FAILED)
    @test !Backend.isfailed(COMPLETED)

    @test Backend.iscompleted(COMPLETED)
    @test Backend.iscompleted(FAILED)
    @test !Backend.iscompleted(PENDING)
    @test !Backend.iscompleted(RUNNING)
end

@testset "Slurm job scripts do not mask the exit status" begin
    backend = ClimaCalibrate.CaltechHPCBackend(;
        directives = [:time => 10, :ntasks => 1],
        modules = String[],
    )
    script = Backend.make_job_script(backend, "julia -e 'error()'")
    # An `exit 0` here would make every failed forward model look successful to
    # the scheduler
    @test !occursin(r"exit\s+0", script)
end
