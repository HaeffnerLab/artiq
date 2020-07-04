using IonSim
using QuantumOptics
using DifferentialEquations
using Distributions

function simulate_with_ion_sim(parameters, pulses, num_ions, b_field)
    #############################################
    # This function must return a dictionary of result values. The keys
    #   must represent each possible state, and the values must
    #   represent the probability of each state.
    # By convention, "S" and "D" are used for the bright and dark states,
    #   respectively. So the keys of the dictionary should look like:
    #   --> for num_ions == 1: "S", "D"
    #           e.g., Dict("S" => 0.1, "D" => 0.9)
    #   --> for num_ions == 2: "SS", "SD", "DS", "DD"
    #           e.g., Dict("SS" => 0.1, "SD" => 0.2, "DS" => 0.3, "DD" => 0.4)
    #   --> for num_ions == 3: "SSS", "SSD", "SDS", "SDD", etc.

    #############################################
    # Create and load the ions
    ions = Array{ca40}(undef, num_ions)
    for i = 1:num_ions
        ions[i] = ca40()
    end
    
    axial_frequency = parameters["TrapFrequencies.axial_frequency"]
    radial_frequency_1 = parameters["TrapFrequencies.radial_frequency_1"]
    radial_frequency_2 = parameters["TrapFrequencies.radial_frequency_2"]
    chain = linearchain(
        ions=ions,
        com_frequencies=(x=radial_frequency_1, y=radial_frequency_2, z=axial_frequency),
        selected_modes=(x=[], y=[], z=[1]))

    #############################################
    # Set up the lasers and the trap
    laser_pulses = []
    for pulse_index in 1:length(pulses)
        pulse = pulses[pulse_index]
        if occursin("729G", pulse["dds_name"])
            push!(laser_pulses, laser(
                label=string(pulse_index),
                k=(x̂ + ẑ)/√2,
                ϵ=(x̂ - ẑ)/√2,
                Δ=pulse["freq"],
            ))
        end
        # TODO: Add 729L1 and 729L2
    end

    lasers = Array{laser}(undef, length(laser_pulses))
    for i in 1:length(laser_pulses)
        lasers[i] = laser_pulses[i]
    end

    ion_trap = trap(configuration=chain, B=b_field*1e-4, Bhat=ẑ, δB=0, lasers=lasers)
    mode = ion_trap.configuration.vibrational_modes.z[1]
    
    #############################################
    # Determine the simulation time when the relevant lasers are on
    simulation_start_time = Inf
    simulation_stop_time = 0
    for laser in lasers
        pulse = pulses[parse(Int, laser.label)]
        simulation_start_time = min(simulation_start_time, pulse["time_on"])
        simulation_stop_time = max(simulation_stop_time, pulse["time_off"])
    end
    if isinf(simulation_start_time)
        simulation_start_time = 0
        simulation_stop_time = 0
    end
    simulation_total_time = simulation_stop_time - simulation_start_time
    simulation_tspan = range(-1e-9, simulation_total_time, length=2)
    
    println("Total simulation time is $(simulation_total_time*1e6) μs")
    println("(start time = $(simulation_start_time*1e6) μs, stop time = $(simulation_stop_time*1e6) μs)")

    #############################################
    # Helpful functions
    function heaviside(t, t0)
        0.5 * (sign(t-t0) + 1)
    end

    function project(solution, states...)
        real.(expect(ion_projector(ion_trap, states...), solution))[2]
    end

    #############################################
    # Set up the time-dependent E-field
    function Ω(t, laser_index)
        sum = 0
        for pulse_index in 1:length(pulses)
            if lasers[laser_index].label == string(pulse_index)
                pulse = pulses[pulse_index]

                intensity_factor = 10^(-0.5) # this corresponds to pi time 3 μs
                pi_min = 3e-6
                t_pi = pi_min * sqrt(intensity_factor / (pulse["amp"] * 10^(-1.5 * pulse["att"] / 10)))
                E = Efield_from_pi_time(t_pi, ion_trap, laser_index, 1, ("S-1/2", "D-1/2"))

                sum += E * heaviside(t, pulse["time_on"] - simulation_start_time)
                sum -= E * heaviside(t, pulse["time_off"] - simulation_start_time)
            end
        end
        return sum
    end
    
    for laser_index in 1:length(lasers)
        lasers[laser_index].E = t -> Ω(t, laser_index)
    end

    #############################################
    # Set up the time-dependent phase
    function ϕ(t, laser_index)
        sum = 0
        for pulse_index in 1:length(pulses)
            if lasers[laser_index].label == string(pulse_index)
                pulse = pulses[pulse_index]
                sum += 2π * pulse["phase"] * heaviside(t, pulse["time_on"] - simulation_start_time)
                sum -= 2π * pulse["phase"] * heaviside(t, pulse["time_off"] - simulation_start_time)
            end
        end
        return sum
    end
    
    for laser_index in 1:length(lasers)
        lasers[laser_index].ϕ = t -> ϕ(t, laser_index)
    end
    
    #############################################
    # Run the simulation
    initial_state = undef
    if length(ions) == 1
        initial_state = ion_state(ion_trap, "S-1/2")
    elseif length(ions) == 2
        initial_state = ion_state(ion_trap, "S-1/2", "S-1/2")
    elseif length(ions) == 3
        initial_state = ion_state(ion_trap, "S-1/2", "S-1/2", "S-1/2")
    end

    simulation_tstops = []
    for laser in lasers
        pulse = pulses[parse(Int, laser.label)]
        append!(simulation_tstops, [
            pulse["time_on"] - simulation_start_time,
            pulse["time_off"] - simulation_start_time,
            ])
    end
    println("Calculated pulse start/stop times as $simulation_tstops")

    h = hamiltonian(ion_trap, timescale=1, rwa_cutoff=1e5) #lamb_dicke_order=1)
    @time tout, solution = timeevolution.schroedinger_dynamic(
        simulation_tspan,
        initial_state ⊗ fockstate(mode.basis, 0),
        h,
        callback=PresetTimeCallback(simulation_tstops, integrator -> return));

    #############################################
    # Measure the expectation values
    result = undef
    s_states = ["S-1/2", "S+1/2"]
    d_states = ["D-5/2", "D-3/2", "D-1/2" ,"D+1/2", "D+3/2", "D+5/2"]
    if length(ions) == 1
        result = Dict(
            "S" => sum([project(solution, state) for state=s_states]),
            "D" => sum([project(solution, state) for state=d_states]),
            )
    elseif length(ions) == 2
        result = Dict(
            "SS" => sum([project(solution, state1, state2) for state1=s_states, state2=s_states]),
            "SD" => sum([project(solution, state1, state2) for state1=s_states, state2=d_states]),
            "DS" => sum([project(solution, state1, state2) for state1=d_states, state2=s_states]),
            "DD" => sum([project(solution, state1, state2) for state1=d_states, state2=d_states]),
            )
    elseif length(ions) == 3
        result = Dict(
            "SSS" => sum([project(solution, state1, state2, state3) for state1=s_states, state2=s_states, state3=s_states]),
            "SSD" => sum([project(solution, state1, state2, state3) for state1=s_states, state2=s_states, state3=d_states]),
            "SDS" => sum([project(solution, state1, state2, state3) for state1=s_states, state2=d_states, state3=s_states]),
            "SDD" => sum([project(solution, state1, state2, state3) for state1=s_states, state2=d_states, state3=d_states]),
            "DSS" => sum([project(solution, state1, state2, state3) for state1=d_states, state2=s_states, state3=s_states]),
            "DSD" => sum([project(solution, state1, state2, state3) for state1=d_states, state2=s_states, state3=d_states]),
            "DDS" => sum([project(solution, state1, state2, state3) for state1=d_states, state2=d_states, state3=s_states]),
            "DDD" => sum([project(solution, state1, state2, state3) for state1=d_states, state2=d_states, state3=d_states]),
            )
    end
    
    #############################################
    # Apply projection noise and renormalize
    total_probability = 0
    for (state, probability) in result
        result[state] = mean(rand(Binomial(1, probability), 100))
        total_probability += result[state]
    end
    for (state, probability) in result
        result[state] = probability / total_probability
    end

    #############################################
    # Return the result
    println("Simulation results: $result")
    return result

end