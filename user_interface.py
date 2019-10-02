


def soil_and_pulses_print(Lx, Lpml):
    Lpml = int(Lpml)
    l = int(0.5 * Lx)
    soil = "_____|"
    soil_patch = "___▄___" # 5 characters
    numbers = "      "
    for i in range(l):
        soil = soil + soil_patch
        numbers = numbers + "   {}   ".format(i)
    soil = soil + "|_____"
    n_chars = len(soil) - 12
    character_length = Lx / n_chars
    print()
    print(soil)
    print(numbers)
    return l, character_length, n_chars

def input_sources(l, Lx, character_length, n_chars):
    i = - Lx / 2 + character_length * 3.5
    source_positions_inputs = str(input("enter position of sources (E.g: 0,1,2): ")).split(",")
    source_positions_inputs = [int(i) for i in source_positions_inputs]
    #print(source_positions_inputs)
    sources_positions = [7 * character_length * j + i  for j in source_positions_inputs]
    return sources_positions

def type_of_medium_input():
    type_of_medium = str(input("Enter type of medium\n"+
                            " Heterogeneous = 0\n"  +
                            " Homomogeneous = 1\n " +
                            " Oblique       = 2\n " +
                            " Choose:  "))
    print(type_of_medium)
    if type_of_medium == "0":
        type_of_medium = "heterogeneous"
    elif type_of_medium == "1":
        type_of_medium = "homogeneous"
    elif type_of_medium == "2":
        type_of_medium = "oblique"
    return type_of_medium



def save_info_oblique(file_name, materials, pulses, t_end, exec_time, used_h, stable_hx, dt, cfl_ct, Lx, Ly, Lpml):
    with open(file_name,"w") as file:
        file.write("MEDIUM PARAMETERS\n")
        for material in materials:
            file.write("\n")
            file.write(material.info())
        file.write("\n\n")

        file.write("MESH\n")
        file.write("Lx [m]= {}\n".format(Lx))
        file.write("Ly [m]= {}\n".format(Ly))
        file.write("Lpml [m]= {}\n".format(Lpml))
        file.write("used hx [m]= {}\n".format(used_h))
        file.write("stable hx [m]= {}\n".format(stable_hx))
        

        file.write("PULSES\n")
        for pulse in pulses:
            file.write("\n")
            file.write(pulse.pulse_info())

        file.write("\nTIME\n")
        file.write("dt [s]: {}\n".format(dt))
        file.write("CFL constant: {}\n".format(cfl_ct))

        file.write("Final time [s]: {}\n".format(t_end))
        file.write("Execution time [min]: {}\n".format(exec_time))

if __name__ == "__main__":

    l, character_l, m_char = soil_and_pulses_print(10, 1)
    input_sources(l, 10, character_l, m_char)
