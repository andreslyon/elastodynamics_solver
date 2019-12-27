import os
import sys
import time
import pickle
import numpy as p
from dolfin.fem.assembling import assemble
from ufl import (as_tensor, Measure, lhs, rhs, inner, grad, exp, tr, Identity)
from dolfin import (MPI, SubDomain, UserExpression, Expression, VectorElement, TensorElement,
                    RectangleMesh, MeshFunction, TestFunctions, TrialFunctions,
                    DOLFIN_EPS, DOLFIN_PI, interpolate, Constant, KrylovSolver,
                    DirichletBC, parameters, set_log_level, LogLevel, Point,
                    MixedElement, FiniteElement, Function, FunctionSpace,
                    XDMFFile, File, plot, TimeSeries)
from user_interface import (soil_and_pulses_print, input_sources, save_info_oblique, type_of_medium_input)
from time_integration_and_compliance import (N_dot, N_ddot, compliance, stable_dt, update, cfl_constant)
from materials import (Material, MaterialFromVelocities, read_materials, cs, cp)
from pulses import (ClassicRickerPulse, ModifiedRickerPulse, Cosine)
from domain_and_layers import (TwoLayeredProperties, Dirichlet,
                               Circle, Layer, ObliqueLayer,
                               ObliqueLayers, MultiLayer,
                               mesh_generator, stable_dx,
                               MaterialProperty)
from pml_functions import alpha_0 as Alpha_0
from pml_functions import alpha_1 as Alpha_1
from pml_functions import alpha_2 as Alpha_2
from pml_functions import beta_0 as Beta_0
from pml_functions import beta_1 as Beta_1
from pml_functions import beta_2 as Beta_2



def forward(mu_expression, lmbda_expression, rho, Lx=10, Ly=10, t_end=1, omega_p=5, amplitude=5000, center=0, target=False):
    Lpml = Lx / 10
    #c_p = cp(mu.vector(), lmbda.vector(), rho)
    max_velocity = 200#c_p.max()

    stable_hx = stable_dx(max_velocity, omega_p)
    nx = int(Lx / stable_hx) + 1
    #nx = max(nx, 60)
    ny = int(Ly * nx / Lx) +1
    mesh = mesh_generator(Lx, Ly, Lpml, nx, ny)
    used_hx = Lx / nx
    dt = stable_dt(used_hx, max_velocity)
    cfl_ct = cfl_constant(max_velocity, dt, used_hx)
    print(used_hx, stable_hx)
    print(cfl_ct)
    #time.sleep(10)
    PE = FunctionSpace(mesh, "DG", 0)
    mu = interpolate(mu_expression, PE)    
    lmbda = interpolate(lmbda_expression, PE)   

    
    m = 2; R = 10e-8; t = 0.0; gamma = 0.50; beta = 0.25

    ff = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    Dirichlet(Lx, Ly, Lpml).mark(ff, 1)

    # Create function spaces
    VE = VectorElement("CG", mesh.ufl_cell(), 1, dim=2)
    TE = TensorElement("DG", mesh.ufl_cell(), 0, shape=(2, 2), symmetry=True)


    W = FunctionSpace(mesh, MixedElement([VE, TE]))
    F = FunctionSpace(mesh, "CG", 2)
    V = W.sub(0).collapse()
    M = W.sub(1).collapse()
    
    alpha_0 = Alpha_0(m, stable_hx, R, Lpml)
    alpha_1 = Alpha_1(alpha_0, Lx, Lpml, degree=2)
    alpha_2 = Alpha_2(alpha_0, Ly, Lpml, degree=2)

    beta_0 = Beta_0(m, max_velocity, R, Lpml)
    beta_1 = Beta_1(beta_0, Lx, Lpml, degree=2)
    beta_2 = Beta_2(beta_0, Ly, Lpml, degree=2)    

    alpha_1 = interpolate(alpha_1, F)
    alpha_2 = interpolate(alpha_2, F)
    beta_1  = interpolate(beta_1, F)
    beta_2  = interpolate(beta_2, F)

    a_ = alpha_1 * alpha_2
    b_ = alpha_1 * beta_2 + alpha_2 * beta_1
    c_ = beta_1 * beta_2

    Lambda_e = as_tensor([[alpha_2, 0],
                         [0, alpha_1]])
    Lambda_p = as_tensor([[beta_2, 0],
                         [0, beta_1]])

    a_ = alpha_1 * alpha_2
    b_ = alpha_1 * beta_2 + alpha_2*beta_1
    c_ = beta_1 * beta_2

    Lambda_e = as_tensor([[alpha_2, 0],
                         [0, alpha_1]])
    Lambda_p = as_tensor([[beta_2, 0],
                         [0, beta_1]])

    # Set up boundary condition
    bc = DirichletBC(W.sub(0), Constant(("0.0", "0.0")), ff, 1)



    # Create measure for the source term
    dx = Measure("dx", domain=mesh)
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

    g = ModifiedRickerPulse(0, omega_p, amplitude, center)

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

    if target:
        xdmffile_u = XDMFFile("inversion_temporal_file/target/u.xdmf")
        pvd  = File("inversion_temporal_file/target/u.pvd")
        xdmffile_u.write(u0, t)
        timeseries_u = TimeSeries("inversion_temporal_file/target/u_timeseries")
    else:
        xdmffile_u = XDMFFile("inversion_temporal_file/obs/u.xdmf")
        xdmffile_u.write(u0, t)
        timeseries_u = TimeSeries("inversion_temporal_file/obs/u_timeseries")
        

    rec_counter = 0

    while t < t_end - 0.5 * dt:
        t += float(dt)

        if rec_counter % 10 == 0:
            print('\n\rtime: {:.3f} (Progress: {:.2f}%)'.format(t, 100 * t / t_end),)

        g.t = t

        # Assemble rhs and apply boundary condition      
        b = assemble(L)
        bc.apply(A, b)
        
        # Compute solution
        solver.solve(S.vector(), b)
        (u, U) = S.split(True)

        # Update previous time step
        update(u, u0, v0, a0, beta, gamma, dt)
        update(U, U0, V0, A0, beta, gamma, dt)

        xdmffile_u.write(u, t)
        pvd << (u,t)
        timeseries_u.store(u.vector(), t)

        energy = inner(u, u) * dx
        E = assemble(energy)
        print("E = ", E)
        print(u.vector().max())




def adjoint(t, mu_expression, lmbda_expression, rho, Lx=10, Ly=10, t_end=1, omega_p=5, amplitude=5000, center=0):
    Lpml = Lx / 10
    #c_p = cp(mu.vector(), lmbda.vector(), rho)
    max_velocity = 150#c_p.max()

    stable_hx = stable_dx(max_velocity, omega_p)
    nx = int(Lx / stable_hx) + 1
    nx = max(nx, 60)
    ny = int(Ly * nx / Lx) +1
    mesh = mesh_generator(Lx, Ly, Lpml, nx, ny)
    used_hx = Lx / nx
    dt = stable_dt(used_hx, max_velocity)
    cfl_ct = cfl_constant(max_velocity, dt, used_hx)


    PE = FunctionSpace(mesh, "DG", 0)
    mu = interpolate(mu_expression, PE)    
    lmbda = interpolate(lmbda_expression, PE)   

    
    m = 2; R = 10e-8; t = 0.0; gamma = 0.50; beta = 0.25

    ff = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    Dirichlet(Lx, Ly, Lpml).mark(ff, 1)

    # Create function spaces
    VE = VectorElement("CG", mesh.ufl_cell(), 1, dim=2)
    TE = TensorElement("DG", mesh.ufl_cell(), 0, shape=(2, 2), symmetry=True)


    W = FunctionSpace(mesh, MixedElement([VE, TE]))
    F = FunctionSpace(mesh, "CG", 2)
    V = W.sub(0).collapse()
    M = W.sub(1).collapse()
    
    alpha_0 = Alpha_0(m, stable_hx, R, Lpml)
    alpha_1 = Alpha_1(alpha_0, Lx, Lpml, degree=2)
    alpha_2 = Alpha_2(alpha_0, Ly, Lpml, degree=2)

    beta_0 = Beta_0(m, max_velocity, R, Lpml)
    beta_1 = Beta_1(beta_0, Lx, Lpml, degree=2)
    beta_2 = Beta_2(beta_0, Ly, Lpml, degree=2)    

    alpha_1 = interpolate(alpha_1, F)
    alpha_2 = interpolate(alpha_2, F)
    beta_1  = interpolate(beta_1, F)
    beta_2  = interpolate(beta_2, F)

    a_ = alpha_1 * alpha_2
    b_ = alpha_1 * beta_2 + alpha_2 * beta_1
    c_ = beta_1 * beta_2

    Lambda_e = as_tensor([[alpha_2, 0],
                         [0, alpha_1]])
    Lambda_p = as_tensor([[beta_2, 0],
                         [0, beta_1]])

    a_ = alpha_1 * alpha_2
    b_ = alpha_1 * beta_2 + alpha_2*beta_1
    c_ = beta_1 * beta_2

    Lambda_e = as_tensor([[alpha_2, 0],
                          [0, alpha_1]])
    Lambda_p = as_tensor([[beta_2, 0],
                          [0, beta_1]])

    # Set up boundary condition
    bc = DirichletBC(W.sub(0), Constant(("0.0", "0.0")), ff, 1)

    # Create measure for the source term
    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=ff)

    # Set up initial values
    u0 = Function(V)
    u0.set_allow_extrapolation(True)
    v0 = Function(V)
    a0 = Function(V)
    U0 = Function(M)
    V0 = Function(M)
    A0 = Function(M)

    u_obs = Function(V)
    u_sta = Function(V)
    load  = Function(V)

    # Test and trial functions
    (theta_u, theta_S) = TrialFunctions(W)
    (w, T) = TestFunctions(W)

    g = ModifiedRickerPulse(0, omega_p, amplitude, center)

    F = rho * inner(a_ * N_ddot(theta_u, u0, a0, v0, dt, beta) - b_ * N_dot(theta_u, u0, v0, a0, dt, beta, gamma) + c_ * theta_u, w) * dx \
        + inner(- N_dot(theta_S, U0, V0, A0, dt, beta, gamma) * Lambda_e + theta_S * Lambda_p, grad(w)) * dx \
        - inner(load, w) * ds \
        + inner(compliance(a_ * N_ddot(theta_S, U0, A0, V0, dt, beta) - b_ * N_dot(theta_S, U0, V0, A0, dt, beta, gamma) + c_ * theta_S, theta_u, mu, lmbda), T) * dx \
        - inner(- Lambda_e * grad(N_dot(theta_u, u0, a0, v0, dt, beta)).T + Lambda_p * grad(theta_u).T, T) * dx 

    a, L = lhs(F), rhs(F)

    # Assemble rhs (once)
    A = assemble(a)

    # Create GMRES Krylov solver
    solver = KrylovSolver(A, "gmres")

    # Create solution function
    S = Function(W)

    rec_counter = 0


    tau = final_time - t
    while t > 0.0 + 0.5 * dt:
        t -= float(dt)
        tau += dt


        if rec_counter % 10 == 0:
            print('\n\rtime: {:.3f} (Progress: {:.2f}%)'.format(t, 100 * t / t_end),)

        g.t = t

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
            xdmffile_u.write(u, t)

        energy = inner(u, u) * dx
        E = assemble(energy)
        print("E = ", E)
        print(u.vector().max())



if __name__ == "__main__":
    mu_expression = Expression("19000000.0", degree=0)
    lmbda_expression = Expression("38000000.0", degree=0)
    rho = 2700
    #materials_data = read_materials()
    #material1 = MaterialFromVelocities(*materials_data[12])
    #lmbda = Constant(material1.lbda)
    #rho = Constant(material1.rho)
    #mu = Constant(material1.mu)

    forward(mu_expression, lmbda_expression, rho, Lx=10, Ly=10, t_end=2, omega_p=5, amplitude=5000, center=0,target=True)
    #timeseries_u = TimeSeries("inversion_temporal_file/target/u_timeseries")
    #print(timeseries_u)
