include("simulate.jl")

test_parameters = Dict(
    "TrapFrequencies.axial_frequency" => 900e3,
    "TrapFrequencies.radial_frequency_1" => 1.3e6,
    "TrapFrequencies.radial_frequency_2" => 1.6e6,
)

num_ions = 1
b_field = 3.799953549
carrier_frequency = 2127405.1614560783

println("Testing a carrier Rabi flop:")
carrier_results = []
for duration in 0:1e-6:20e-6
    test_pulses = [
        Dict(
            "amp" => 1.0,
            "att" => 6.0,
            "dds_name" => "729G",
            "freq" => carrier_frequency,
            "phase" => 0.0,
            "time_off" => duration,
            "time_on" => 0.0
        ),
    ]
    
    result = simulate_with_ion_sim(test_parameters, test_pulses, num_ions, b_field)
    push!(carrier_results, result["D"])
end

println("Testing a blue sideband Rabi flop:")
sideband_results = []
for duration in 0:2e-6:50e-6
    test_pulses = [
        Dict(
            "amp" => 1.0,
            "att" => 6.0,
            "dds_name" => "729G",
            "freq" => carrier_frequency + test_parameters["TrapFrequencies.axial_frequency"],
            "phase" => 0.0,
            "time_off" => duration,
            "time_on" => 0.0
        ),
    ]
    
    result = simulate_with_ion_sim(test_parameters, test_pulses, num_ions, b_field)
    push!(sideband_results, result["D"])
end

println("Carrier results: $carrier_results")
println("Sideband results: $sideband_results")
