using ClimaCalibrate, ClimaAnalysis, OrderedCollections, Test
import Random: MersenneTwister

lon = collect(range(-179.5, 179.5, 360))
lat = collect(range(-89.5, 89.5, 180))
time = collect(range(0.0, 10.0, 10))
data = rand(MersenneTwister(1234), length(lat), length(time), length(lon))
dims = OrderedDict(["lat" => lat, "time" => time, "lon" => lon])
attribs = Dict("long_name" => "hi")
dim_attribs = OrderedDict([
    "lat" => Dict("units" => "deg"),
    "time" => Dict("units" => "days"),
    "lon" => Dict("units" => "deg"),
])
var = ClimaAnalysis.OutputVar(attribs, dims, dim_attribs, data)

@testset "ClimaAnalysisExt" begin
    @test !isnothing(ClimaCalibrate.construct_noise_covariance(var))
end
