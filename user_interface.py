


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
    source_positions_inputs = str(input("enter position of sources: ")).split(",")
    source_positions_inputs = [int(i) for i in source_positions_inputs]
    #print(source_positions_inputs)
    sources_positions = [7 * character_length * j + i  for j in source_positions_inputs]
    return sources_positions


def save_info_oblique(file_name, materials, pulses, t_end, exec_time, h, dt, Lx, Ly, Lpml):
    with open() as file:
        file.write("MEDIUM PARAMETERS\n")
        for material in materials:
            file.write("\n")
            file.write(material.info())
        file.write("\n\n")

        file.write("PULSES\n")
        file.write("Final time [s]: {}\n".format(t_end))
        file.write("Execution time [s]: {}\n".format(exec_time))

if __name__ == "__main__":

    l, character_l, m_char = soil_and_pulses_print(10, 1)
    input_sources(l, 10, character_l, m_char)
