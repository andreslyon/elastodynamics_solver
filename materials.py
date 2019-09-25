import numpy as p


class Material:
    def __init__(self, name, nu, E, rho):
        self.name = name
        self.nu = nu
        self.E = E
        self.rho = rho
        self.lbda = self.nu * self.E / (( 1 + self.nu) * (1 - 2 * self.nu))
        self.mu = 0.5 * self.E / (1 + self.nu)
        self.c_p = p.sqrt(self.lbda + 2 * self.mu) / p.sqrt(self.rho)
        self.c_s = p.sqrt(self.mu / self.rho)
    def print(self):
        print("Material properties of {}".format(self.name))
        print("lambda = {}".format(self.lbda))
        print("mu = {}".format(self.mu))
        print("Cp = {}".format(self.c_p))
        print("Cs = {}".format(self.c_s))
    def info(self):
        return "Material properties of {}".format(self.name) + "\n" \
         + "lambda = {}\n".format(self.lbda) +\
        "mu = {}\n".format(self.mu) + \
        "Cp = {}\n".format(self.c_p) + \
        "Cs = {}\n".format(self.c_s)


class MaterialFromVelocities:
    def __init__(self, name, c_p, c_s, nu, rho):
        self.name = name
        self.c_s = c_s
        self.c_p = c_p
        self.lbda = rho * (c_p ** 2 - 2 * c_s ** 2)
        self.mu = rho * c_s ** 2
        self.rho = rho

    def print(self):
        print("Material properties")
        print("lambda = {}".format(self.lbda))
        print("mu = {}".format(self.mu))
        print("Cp = {}".format(self.c_p))
        print("Cs = {}".format(self.c_s))

    def info(self):
        return "Material properties of {}".format(self.name) + "\n" + "lambda = {}\n".format(self.lbda) +\
        "mu = {}\n".format(self.mu) + \
        "Cp = {}\n".format(self.c_p) + \
        "Cs = {}\n".format(self.c_s)


def read_materials():
    with open("material_data.txt") as file:
        data = file.readlines()
        data = [x.strip("\n") for x in data]
        for i in range(len(data)):
            data[i] = data[i].split(",")
            for j in range(1,len(data[i])):
                data[i][j] = float(data[i][j])
    return data

def print_materials():
    with open("material_data.txt") as file:
        data = file.readlines()
        data = [x.strip("\n") for x in data]
        for i in range(len(data)):
            data[i] = data[i].split(",")
            for j in range(1,len(data[i])):
                data[i][j] = float(data[i][j])
    print()
    print("   id     |   Material          |    Cp    |    Cs     |    nu   ")
    
    for mat,i in zip(data,range(len(data))):
        print("{}      {}         {}           {}         {}".format(i,mat[0],mat[1],mat[2],mat[3]))

