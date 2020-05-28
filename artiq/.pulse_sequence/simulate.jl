using IonSim

function simulate_with_ion_sim(parameters, pulses, num_ions, b_field)
    # This function must return a dictionary of result values.
    # Typically, this dictionary will represent the probability
    #    of each possible readout state, and so each value will
    #    be between 0.0 and 1.0.
    # e.g.,  for one ion: Dict("dark" => 0.1)
    # e.g.,           or: Dict("S" => 0.1, "D" => 0.9)
    # e.g., for two ions: Dict("ion 0 dark" => 0.1, "ion 1 dark" => 0.5)
    # e.g.,           or: Dict("SS" => 0.1, "SD" => 0.2, "DS" => 0.3, "DD" => 0.4)
    # The names of each readout state are not critical. They are used only for
    #    display in the grapher and output to the results file.

    ions = []
    for i = 1:num_ions
        push!(ions, ca40())
    end

    # TODO: If necessary, read parameters here. For example:
    axial_frequency = parameters["TrapFrequencies.axial_frequency"]

    for pulse in pulses
        dds_name = pulse["dds_name"]
        time_on = pulse["time_on"]
        time_off = pulse["time_off"]
        freq = pulse["freq"]
        amp = pulse["amp"]
        att = pulse["att"]
        phase = pulse["phase"]

        # TODO: Do something with this pulse.
    end

    # TODO: Run the simulation and return the result.

    return Dict("S" => 0.1, "D" => 0.9) # fake single-ion result indicating P(S)=0.1, P(D)=0.9

end