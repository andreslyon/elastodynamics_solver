import os
import sys
from dolfin import (MPI, SubDomain, UserExpression, VectorElement, TensorElement,
                    MixedElement, FiniteElement, Function, FunctionSpace,
                    RectangleMesh, MeshFunction, TestFunctions, TrialFunctions,
                    DirichletBC, parameters, set_log_level, LogLevel, Point,
                    DOLFIN_EPS, DOLFIN_PI, interpolate, Constant, KrylovSolver,
                    XDMFFile, File, plot)
from dolfin.fem.assembling import assemble
from ufl import (as_tensor, Measure, lhs, rhs, inner, grad, exp, tr, Identity)
from time_integration_and_compliance import (N_dot, N_ddot, compliance, stable_dt, update, cfl_constant)
from domain_and_layers import (mesh_generator, stable_dx, ObliqueLayers, MultiLayer, TwoLayeredProperties, Dirichlet,
                               Circle, Layer, MaterialProperty)
from materials import (Material, MaterialFromVelocities, read_materials, print_materials)
from user_interface import (soil_and_pulses_print, input_sources, save_info_oblique, type_of_medium_input)
from pml_functions import (alpha_0, alpha_1, alpha_2, beta_0, beta_1, beta_2)
from pulses import (ClassicRickerPulse, ModifiedRickerPulse, Cosine)
import numpy as p
import time
import pickle


if __name__ == "__main__":
        
    Lx = float(input("Enter Lx [m]: "))
    Ly = float(input("Enter Ly ( < Lx ) [m]: "))
    Lx_ly_prop = Lx / Ly
    Lpml = Lx / 10
    n_sources, character_length, m_char = soil_and_pulses_print(Lx, Lpml)
    sources_positions = input_sources(n_sources, Lx, character_length, m_char)

    omega_p_list = []
    amplitude_list = []

    for i in range(len(sources_positions)):
        amplitude_list.append(float(input("Amplitude of source at {} [m]: ".format(sources_positions[i]))))
        omega_p_list.append(float(input("Peak freq. of source at {} [Hz]: ".format(sources_positions[i]))))
    
    max_omega_p = max(omega_p_list)
    
    # ================= MESH  ================= #

    type_of_medium = type_of_medium_input()
    materials_data = read_materials()
    print_materials()
    materials = []


    if (type_of_medium == "oblique"):
        for _ in range(3):
            material_id = int(input("Enter id of material: "))
            materials.append(MaterialFromVelocities(*materials_data[material_id]))
    
        lambdas = [material.lbda for material in materials]
        rhos = [material.rho for material in materials]
        mus =  [material.mu for material in materials]

        lmbda = ObliqueLayers(lambdas)
        rho = ObliqueLayers(rhos)
        mu = ObliqueLayers(mus)

    elif type_of_medium == "homogeneous":
        material_id = int(input("Enter id of material: "))
        materials.append(MaterialFromVelocities(*materials_data[material_id]))
        lmbda = Constant(materials[0].lbda)
        rho = Constant(materials[0].rho)
        mu = Constant(materials[0].mu)

    elif (type_of_medium == "heterogeneous"):
        for _ in range(2):
            material_id = int(input("Enter id of material: "))
            materials.append(MaterialFromVelocities(*materials_data[material_id]))
    
        lambdas = [material.lbda for material in materials]
        rhos = [material.rho for material in materials]
        mus =  [material.mu for material in materials]


    t_end   = float(input("Enter final time [s]: "))
    max_velocity = max([material.c_p for material in materials])
    min_velocity = min([material.c_p for material in materials])
    max_omega_p = max(omega_p_list)

    stable_hx = stable_dx(max_velocity, max_omega_p)
    nx = int(Lx / stable_hx) + 1
    nx = max(nx, 60)
    ny = int(Ly * nx / Lx) +1
    mesh = mesh_generator(Lx, Ly, Lpml, nx, ny)
    used_hx = Lx / nx
    dt      =  stable_dt(used_hx, max_velocity)
    cfl_ct = cfl_constant(max_velocity, dt, used_hx)

    V_r = max_velocity
    car_dim = stable_hx
    m = 2 
    R = 10e-8
    t       = 0.0
    gamma   = 0.50
    beta    = 0.25

    # ====== SUBDOMAIn FOR MEDIUM TYPE ========= #
    if type_of_medium == "heterogeneous":
        layer_start_0 = 0
        layer_start_1 = layer_end_0 = Ly / 2
        layer_end_1 = Ly + Lpml

        subdomain_0 = Layer(Lx, Ly, Lpml, layer_start_0, layer_end_0)
        subdomain_1 = Layer(Lx, Ly, Lpml, layer_start_1, layer_end_1)
        subdomains = MeshFunction("size_t", mesh, 2, 0)
        subdomain_0.mark(subdomains, 0)
        subdomain_1.mark(subdomains, 1)


        lmbda = MaterialProperty(lambdas[0], lambdas[1], subdomains=subdomains,degree=0)
        rho = MaterialProperty(rhos[0], rhos[1], subdomains=subdomains,degree=0)
        mu =  MaterialProperty(mus[0], mus[1], subdomains=subdomains,degree=0)
        
        # MEASURE
        dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
    else:
        dx = Measure("dx", domain=mesh)


  # Optimization options for the form compiler
    parameters["krylov_solver"]["maximum_iterations"] = 300
    parameters["krylov_solver"]["relative_tolerance"] = 1.0e-10
    parameters["krylov_solver"]["absolute_tolerance"] = 1.0e-10

  # ================= MPI PARAMETERS  ================= #

    # MPI Parameters
    comm = MPI.comm_world
    rank = MPI.rank(comm)

    # Set log level for parallel
    set_log_level(LogLevel.ERROR)
    if rank == 0:
        set_log_level(LogLevel.PROGRESS)

    ff = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    Dirichlet(Lx, Ly, Lpml).mark(ff, 1)

    # Create function spaces
    VE = VectorElement("CG", mesh.ufl_cell(), 1, dim=2)
    TE = TensorElement("DG", mesh.ufl_cell(), 0, shape=(2, 2), symmetry=True)

    W = FunctionSpace(mesh, MixedElement([VE, TE]))
    F = FunctionSpace(mesh, "CG", 2)
    V = W.sub(0).collapse()
    M = W.sub(1).collapse()
    

    alpha_0 = alpha_0(m, car_dim, R, Lpml)
    alpha_1 = alpha_1(alpha_0, Lx, Lpml, degree=2)
    alpha_2 = alpha_2(alpha_0, Ly, Lpml, degree=2)

    beta_0 = beta_0(m, V_r, R, Lpml)
    beta_1 = beta_1(beta_0, Lx, Lpml, degree=2)
    beta_2 = beta_2(beta_0, Ly, Lpml, degree=2)    

    alpha_1 = interpolate(alpha_1, F)
    alpha_2 = interpolate(alpha_2, F)
    beta_1  = interpolate(beta_1, F)
    beta_2  = interpolate(beta_2, F)

    a_ = alpha_1 * alpha_2
    b_ = alpha_1 * beta_2 + alpha_2 * beta_1
    c_ = beta_1 * beta_2

    Lambda_e = as_tensor([[alpha_2, 0], [0, alpha_1]])
    Lambda_p = as_tensor([[beta_2, 0], [0, beta_1]])

    a_ = alpha_1 * alpha_2
    b_ = alpha_1 * beta_2 + alpha_2*beta_1
    c_ = beta_1 * beta_2

    Lambda_e = as_tensor([[alpha_2, 0], [0, alpha_1]])
    Lambda_p = as_tensor([[beta_2, 0], [0, beta_1]])

    # Set up boundary condition
    bc = DirichletBC(W.sub(0), Constant(("0.0", "0.0")), ff, 1)

    # Create measure for the source term
    ds = Measure("ds", domain=mesh, subdomain_data=ff)

    # Set up initial values
    u0 = Function(V)
    u0.set_allow_extrapolation(True)
    v0 = Function(V)
    a0 = Function(V)
    U0 = Function(M)
    V0 = Function(M)
    A0 = Function(M)

    # Test and trial functions
    (u, S) = TrialFunctions(W)
    (w, T) = TestFunctions(W)

    pulses = [ModifiedRickerPulse(t,omega_p_list[i], amplitude_list[i], center=sources_positions[i]) for i in range(len(omega_p_list))] 
    
    g = sum(pulses)

    F = rho * inner(a_ * N_ddot(u, u0, a0, v0, dt, beta) \
        + b_ * N_dot(u, u0, v0, a0, dt, beta, gamma) + c_ * u, w) * dx \
        + inner(N_dot(S, U0, V0, A0, dt, beta, gamma).T * Lambda_e + S.T * Lambda_p, grad(w)) * dx \
        - inner(g, w) * ds \
        + inner(compliance(a_ * N_ddot(S, U0, A0, V0, dt, beta) + b_ * N_dot(S, U0, V0, A0, dt, beta, gamma) + c_ * S, u, mu, lmbda), T) * dx \
        - 0.5 * inner(grad(u) * Lambda_p + Lambda_p * grad(u).T + grad(N_dot(u, u0, v0, a0, dt, beta, gamma)) * Lambda_e \
        + Lambda_e * grad(N_dot(u, u0, v0, a0, dt, beta, gamma)).T, T) * dx \


    a, L = lhs(F), rhs(F)

    # Assemble rhs (once)
    A = assemble(a)

    # Create GMRES Krylov solver
    solver = KrylovSolver(A, "gmres")

    # Create solution function
    S = Function(W)

    experiment_count_file = open("experiment_counter",'rb')
    experiment_count = pickle.load(experiment_count_file)
    experiment_count_file.close()

    paraview_file_name = "experiment_{}".format(experiment_count)
    info_file_name = "{}_experiments_info/info_n{}.txt".format(type_of_medium,experiment_count)

    experiment_count += 1
    experiment_count_file = open("experiment_counter",'wb')

    pickle.dump(experiment_count,experiment_count_file)
    experiment_count_file.close()

    
    xdmf_file = XDMFFile(mesh.mpi_comm(), "output/elastodynamics.xdmf")
    pvd  = File("paraview_{}/{}.pvd".format(type_of_medium,paraview_file_name))
    pvd << (u0, t)
    rec_counter = 0

    t0 = time.time()
    while t < t_end - 0.5 * dt:
        t += float(dt)

        if rank == 0 and rec_counter % 10 == 0:
            print('\n\rtime: {:.3f} (Progress: {:.2f}%)'.format(t, 100 * t / t_end),)
            print("time taken til this point: {}".format(time.time()-t0))

        for pulse in pulses:
            pulse.t = t
        g = sum(pulses)

        # Assemble rhs and apply boundary condition      
        b = assemble(L)
        bc.apply(A, b)
        
        # Compute solution
        solver.solve(S.vector(), b)
        (u, U) = S.split(True)

        # Update previous time step
        update(u, u0, v0, a0, beta, gamma, dt)
        update(U, U0, V0, A0, beta, gamma, dt)

        if rec_counter % 2 == 0:
            pvd << (u0, t)

        rec_counter += 1
        energy = inner(u, u) * dx
        E = assemble(energy)
        print("E = ", E)
    t_f = (time.time() - t0) / 60

    save_info_oblique(info_file_name, materials, pulses, 
                      t_end, t_f, used_hx, stable_hx, dt, cfl_ct, Lx, Ly, Lpml)
    print('\007')
    print('\007')
    print('\007')
        


       
