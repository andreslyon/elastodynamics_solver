# Copyright (C) 2019 Hernan Mella
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

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
import numpy as p
#from analysis_functions import *
import physical_parameters 
import time

###########################


class ObliqueLayers(UserExpression):
  def __init__(self, layers_properties, **kwargs):
      super().__init__(**kwargs)
      self.layers_properties = layers_properties

  def eval(self, value, x):
    tol = 10e-14
    if (x[1] >= -x[0] / 3 - 2 + tol) and (x[1] <= x[0] - 3 + tol):
      value[0] = self.layers_properties[0]
    elif (x[1] >= -x[0] / 3 - 2 + tol) and (x[1] >= x[0] - 3 + tol):
      value[0] = self.layers_properties[1]
    else:
      value[0] = self.layers_properties[2]

class multilayer(UserExpression):
  def __init__(self, layer_properties, layers_depth, **kwargs):
      super().__init__(**kwargs)
      self.layer_properties = layer_properties
      self.layers_depth = layers_depth

  def eval(self, value, x):
    tol = 1e-14

    for i in range(1, len(self.layers_depth)):
 
      if abs(self.layers_depth[i - 1]) + tol <= abs(x[1]) and abs(x[1]) <= abs(self.layers_depth[i]) + tol:

          value[0] = self.layer_properties[i - 1]


class Lmbda(UserExpression):
    def __init__(self, lmbda_0, lmbda_1, first_layer_depth, **kwargs):
        super().__init__(**kwargs)
        self.lmbda_0 = lmbda_0
        self.lmbda_1 = lmbda_1
        self.first_layer_depth = first_layer_depth

    def eval(self, value, x):
        tol = 1e-14
        if abs(x[1]) <= abs(first_layer_depth) + tol:
            value[0] = self.lmbda_0
        else:
            value[0] = self.lmbda_1


class Mu(UserExpression):
    def __init__(self, mu_0, mu_1, first_layer_depth, **kwargs):
        super().__init__(**kwargs)
        self.mu_0 = mu_0
        self.mu_1 = mu_1
        self.first_layer_depth = first_layer_depth

    def eval(self, value, x):
        tol = 1e-14
        if abs(x[1]) <= abs(first_layer_depth) + tol:
            value[0] = self.mu_0
        else:
            value[0] = self.mu_1
        

class Rho(UserExpression):
    def __init__(self, rho_0, rho_1, first_layer_depth, **kwargs):
        super().__init__(**kwargs)
        self.rho_0 = rho_0
        self.rho_1 = rho_1
        self.first_layer_depth = first_layer_depth

    def eval(self, value, x):
        tol = 1e-14
        if abs(x[1]) <= abs(first_layer_depth) + tol:
            value[0] = self.rho_0
        else:
            value[0] = self.rho_1
        

# Update previous time step using Newmark scheme
def update(u, u0, v0, a0, beta, gamma, dt):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec, u0_vec  = u.vector(), u0.vector()
    v0_vec, a0_vec = v0.vector(), a0.vector()

    # Update acceleration and velocity

    # a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
    a_vec = (1.0/(2.0*beta))*( (u_vec - u0_vec - v0_vec*dt)/(0.5*dt*dt) - (1.0-2.0*beta)*a0_vec )

    # v = dt * ((1-gamma)*a0 + gamma*a) + v0
    v_vec = dt*((1.0 - gamma)*a0_vec + gamma*a_vec) + v0_vec

    # Update (u0 <- u0)
    v0.vector()[:], a0.vector()[:] = v_vec, a_vec
    u0.vector()[:] = u.vector()

# Dirichlet boundary
class Dirichlet(SubDomain):
  def inside(self, x, on_boundary):
    return x[0] <= -0.5*Lx-Lpml + DOLFIN_EPS \
        or x[0] >= 0.5*Lx+Lpml-DOLFIN_EPS \
        or x[1] <= -Ly-Lpml + DOLFIN_EPS and on_boundary


# Source domain
class circle(SubDomain):
  def inside(self, x, on_boundary):
    return pow(x[0] - xc, 2) + pow(x[1] - yc, 2) <= pow(rd, 2)


# ================= PULSES  ================= #
# Explosive Ricker source
class ricker_pulse(UserExpression):
    def __init__(self, t, omega, **kwargs):
        self.t     = t
        self.omega = omega
        super().__init__(**kwargs)
  
  def eval(self, values, x):
    # Ricker pulse Parameters
    r  = ((x[0] - xc)**2 + (x[1] - yc)**2)**0.5
    
    u  = self.omega*self.t - 3*pow(6,0.5)
    Sp = (1.0 - (r / rd) ** 2) ** 3

    #if self.t <= 6 * pow(6, 0.5) / self.omega:

    Tp = ((0.25 * pow(u, 2) - 0.5) * exp(-0.25 * pow(u, 2)) \
         -13 * exp(-13.5)) / (0.5 + 13 * exp(-13.5))
    #else:
   #  Tp = 0

    values[0] = (5e4)*physical_parameters.amplitude * Tp * Sp * (x[0] - xc) / r
    values[1] = (5e4)*physical_parameters.amplitude * Tp * Sp * (x[1] - yc) / r

  def value_shape(self):
    return (2,)


# Surface Ricker source
class surface_ricker_pulse(UserExpression):
  def __init__(self, t, omega, center=0, **kwargs):
    super().__init__(**kwargs)
    self.t  = t
    self.omega  = omega
    self.center = center

  def eval(self, values, x):

    u  = self.omega*self.t - 3*pow(6,0.5)
    if abs(x[0] - self.center) < 0.25:
      Tp = ((0.25*pow(u,2) - 0.5)*exp(-0.25*pow(u,2)) - 13*exp(-13.5))/(0.5 + 13*exp(-13.5))
      values[0] = 0.0
      values[1] = 5000*Tp   # [KPa]
    else:
      values[0] = 0.0
      values[1] = 0.0      #5000*Tp 


  def value_shape(self):
    return (2,)


class ClassicRickerPulse(UserExpression):
    def __init__(self, t, omega, center=0, **kwargs):
        super().__init__(**kwargs)
        self.t  = t
        self.omega  = omega * 2 * p.pi # rads/s
        self.center = center

  def eval(self, values, x):
    if abs(x[0] - self.center) < 0.25:
      values[0] = 0
      values[1] = 5000 * (1 - 0.5 * self.omega ** 2 * self.t ** 2) * exp( - 0.25 *  self.omega ** 2 * self.t ** 2) 
    else:
      values[0] = 0.0
      values[1] = 0.0      #5000*Tp 


  def value_shape(self):
    return (2,)
# ================= PML PARAMETERS ================= #

class alpha_1(UserExpression):
    def __init__(self, alpha_0, Lx, Lpml, **kwargs):
        super().__init__(**kwargs)
        self.alpha_0 = alpha_0
        self.Lx = Lx
        self.Lpml = Lpml

  def eval(self, values, x, **kwargs):
        values[0] = 1.0 + self.alpha_0*pow((abs(x[0]) - 0.5 * self.Lx + abs(abs(x[0]) - 0.5 * self.Lx )) / (2 * self.Lpml), 2)

class alpha_2(UserExpression):
    def __init__(self, alpha_0, Ly, Lpml, **kwargs):
      super().__init__(**kwargs)
      self.alpha_0 = alpha_0
      self.Ly = Ly
      self.Lpml = Lpml
    def eval(self, values, x, **kwargs):
      values[0] = 1.0 + self.alpha_0 * pow((-(x[1] + self.Ly) + abs(x[1] + self.Ly))/(2*self.Lpml), 2)

class beta_1(UserExpression):
  def __init__(self, beta_0, Lx, Lpml, **kwargs):
    super().__init__(**kwargs)
    self.beta_0 = beta_0
    self.Lx = Lx
    self.Lpml = Lpml
  
  def eval(self, values, x, **kwargs):
    values[0] = self.beta_0*pow((abs(x[0]) - 0.5 * self.Lx + abs(abs(x[0]) - 0.5 * self.Lx))/(2 * self.Lpml), 2)

class beta_2(UserExpression):
  def __init__(self, beta_0, Ly, Lpml, **kwargs):
    super().__init__(**kwargs)
    self.beta_0 = beta_0
    self.Ly = Ly
    self.Lpml = Lpml  
  def eval(self, values, x, **kwargs):
    values[0] = self.beta_0 * pow((-(x[1] + Ly) + abs(x[1] + self.Ly))/(2 * self.Lpml), 2)

# ================= TIME InTEGRATIOn ================= #

# Newmark scheme for time integration
def N_ddot(arg):
  """ Acceleration """
  # Extract shape
  shape = arg.ufl_shape
  if shape == (2,):
    output = (arg - u0 - dt*v0)/(beta*dt*dt) - a0*(1 - 2*beta)/(2*beta)
  elif shape == (2, 2):
    output = (arg - U0 - dt*V0)/(beta*dt*dt) - A0*(1 - 2*beta)/(2*beta)
  return output

def N_dot(arg):
  """ Velocity """
  # Extract shape
  shape = arg.ufl_shape
  if shape == (2,):
    output = gamma*(arg - u0)/(beta*dt) - v0*(gamma - beta)/beta - a0*dt*(gamma - 2*beta)/(2*beta)
  elif shape == (2, 2):
    output = gamma*(arg - U0)/(beta*dt) - V0*(gamma - beta)/beta - A0*dt*(gamma - 2*beta)/(2*beta)
  return output

# Compliance operator
def compliance(sigma):
  ' Returns the strain tensor as a function of sigma '
  return sigma/(2*mu) - lmbda/(4*mu*(lmbda + mu))*tr(sigma)*Identity(u.geometric_dimension())

if __name__ == "__main__":
    
    experiment = sys.argv[1]

    print("record: True/False")
    record = True#input()

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

    # ================= GEOMETRY PARAMETERS  ================= #

    rd   = 0.4    # disk radius
    xc   = 0.0    # x-coordinate of the center of the disk
    yc   = -5.0   # y-coordinate of the center of the disk
    Lpml = 1.0    # width of the PML layer
    Lx   = 10.0   # width of the interior domain
    Ly   = 10.0   # heigth of the interior domain

        
    # Attenuation functions
    beta_0  = physical_parameters.beta_0    # 280  # propagation
    alpha_0 = physical_parameters.alpha_0   # 2.76 # evanescent
        
    # Mesh generation and refinement
    nx = 80
    ny = 80
    mesh = RectangleMesh(Point(- 0.5 * Lx - Lpml, 0.0), Point(0.5 * Lx + Lpml, - Ly - Lpml), nx, ny, "crossed")

    # Markers for Dirichlet bc
    ff = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    Dirichlet().mark(ff, 1)

    # Markers for disk source
    mf = MeshFunction("size_t", mesh, mesh.geometry().dim())
    circle().mark(mf, 1)

    # Create function spaces
    VE = VectorElement("CG", mesh.ufl_cell(), 1, dim=2)
    TE = TensorElement("DG", mesh.ufl_cell(), 0, shape=(2, 2), symmetry=True)

    W = FunctionSpace(mesh, MixedElement([VE, TE]))
    F = FunctionSpace(mesh, "CG", 2)
    V = W.sub(0).collapse()
    M = W.sub(1).collapse()

    # Time stepping Parameters
    omega_p = 8*DOLFIN_PI
    dt      = 0.0025
    t       = 0.0
    t_end   = 1
    gamma   = 0.50
    beta    = 0.25


    # Circle CI  test
    if "CI" in os.environ.keys():
        T = 3 * dt
    else:
        T = 1400 * dt


    # Elasticity Parameters

    str_exp = "Experiment"
    if experiment == "4":
      first_layer_depth = - Ly / 3
      r_0, r_1 = physical_parameters.rho, physical_parameters.rho
      l_0, l_1 = physical_parameters.lbda / 0.7, physical_parameters.lbda * 10
      m_0, m_1 = physical_parameters.mu, physical_parameters.mu 
      g = surface_ricker_pulse(t=0.0, omega=omega_p, center=0, degree=1) + surface_ricker_pulse(t=0.0, omega=omega_p, center=2, degree=1)


      file_name = "{}_{}".format(experiment, str_exp)



    elif experiment == "1-1":
      first_layer_depth = - Ly / 3
      r_0, r_1 = physical_parameters.rho, physical_parameters.rho
      l_0, l_1 = physical_parameters.lbda, physical_parameters.lbda 
      m_0, m_1 = physical_parameters.mu, physical_parameters.mu       
      g = ClassicRickerPulse(t=0.0, omega=omega_p, center=0, degree=1)

      file_name = "{}_{}".format(experiment, str_exp)

    elif experiment == "2":
      first_layer_depth = - Ly / 3
      r_0, r_1 = physical_parameters.rho / 2, physical_parameters.rho
      l_0, l_1 = physical_parameters.lbda, physical_parameters.lbda 
      m_0, m_1 = physical_parameters.mu, physical_parameters.mu      
      g = surface_ricker_pulse(t=0.0, omega=omega_p, center=0, degree=1)

      file_name = "{}_{}".format(experiment, str_exp)
    

    elif experiment == "3":
      first_layer_depth = - Ly / 3
      r_0, r_1 = physical_parameters.rho, physical_parameters.rho
      l_0, l_1 = physical_parameters.lbda, physical_parameters.lbda 
      m_0, m_1 = physical_parameters.mu, physical_parameters.mu * 2
      g = surface_ricker_pulse(t=0.0, omega=omega_p, center=0, degree=1)

      file_name = "{}_{}".format(experiment, str_exp)
    

    elif experiment == "5":
      first_layer_depth = - Ly / 3
      r_0, r_1 = physical_parameters.rho, physical_parameters.rho
      l_0, l_1 = physical_parameters.lbda, physical_parameters.lbda 
      m_0, m_1 = physical_parameters.mu, physical_parameters.mu       
      g = surface_ricker_pulse(t=0.0, omega=omega_p, center = 2.5, degree=1)

      file_name = "{}_{}".format(experiment, str_exp)


    elif experiment == "8":
      layers_depth = [0, 1.5, 3.5, 4.5, 6, 7.5, Ly + Lpml]
      lambdas = [physical_parameters.lbda,
                 physical_parameters.lbda * 20,
                 physical_parameters.lbda,
                 physical_parameters.lbda * 20,
                 physical_parameters.lbda,
                 physical_parameters.lbda * 20]
      mus = [physical_parameters.mu for _ in range(len(lambdas))]
      rho = Constant(2200)
      lmbda = multilayer(lambdas, layers_depth)
      mu =  multilayer(mus, layers_depth)

      g = surface_ricker_pulse(t=0.0, omega=omega_p, center = 0, degree=1)

      file_name = "{}_{}".format(experiment, str_exp)
    
    elif experiment == "12":
      lambdas = [physical_parameters.test_material.lbda, physical_parameters.test_material.lbda*20, physical_parameters.test_material.lbda/10]
      mus = [physical_parameters.test_material.mu, physical_parameters.test_material.mu, physical_parameters.test_material.mu]
      rhos = [physical_parameters.test_material.rho, physical_parameters.test_material.rho, physical_parameters.test_material.rho]
      lmbda = ObliqueLayers(lambdas)
      mu = ObliqueLayers(mus)
      rho = ObliqueLayers(rhos)

      g = surface_ricker_pulse(t=0.0, omega=omega_p, center = 0, degree=1)
      file_name = "{}_{}".format(experiment, str_exp)


    if experiment != "12":
      rho   = Rho(r_0, r_1, first_layer_depth)

      lmbda = Lmbda(l_0, l_1, first_layer_depth) # Constant(7.428e+04)
      mu    = Mu(m_0, m_1, first_layer_depth)   # Constant(7.428e+04)

    # Attenuation and stretching functions and tensors
    alpha_1 = interpolate(alpha_1(degree=2), F)
    alpha_2 = interpolate(alpha_2(degree=2), F)
    beta_1  = interpolate(beta_1(degree=2), F)
    beta_2  = interpolate(beta_2(degree=2), F)

    a_ = alpha_1*alpha_2
    b_ = alpha_1*beta_2 + alpha_2*beta_1
    c_ = beta_1*beta_2

    Lambda_e = as_tensor([[alpha_2, 0],[0, alpha_1]])
    Lambda_p = as_tensor([[beta_2, 0],[0, beta_1]])

    # Set up boundary condition
    bc = DirichletBC(W.sub(0), Constant(("0.0", "0.0")), ff, 1)

    # Create measure for the source term
    dx = Measure("dx", domain=mesh, subdomain_data=mf)
    ds = Measure("ds", subdomain_data=ff, domain=mesh)

    # Source term
    #f = ricker_pulse(t=0.0, omega=omega_p, degree=1)
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


    # Define variational problem
    F = rho*inner(a_*N_ddot(u) + b_*N_dot(u) + c_*u, w)*dx \
        + inner(N_dot(S).T*Lambda_e + S.T*Lambda_p, grad(w))*dx \
        - inner(g, w)*ds \
        + inner(compliance(a_*N_ddot(S) + b_*N_dot(S) + c_*S), T)*dx \
        - 0.5*inner(grad(u)*Lambda_p + Lambda_p*grad(u).T + grad(N_dot(u))*Lambda_e \
        + Lambda_e*grad(N_dot(u)).T, T)*dx \
        #- inner(f, w)*dx(1) \


    a, L = lhs(F), rhs(F)

    # Assemble rhs (once)
    A = assemble(a)

    # Create GMRES Krylov solver
    solver = KrylovSolver(A, "gmres")

    # Create solution function
    S = Function(W)

    # Solving loop
    xdmf_file = XDMFFile(mesh.mpi_comm(), "output/elastodynamics.xdmf")

    if record:
        pvd  = File("paraview/{}.pvd".format(file_name))
        pvd << (u0, t)


    # =========== TIME TO PML =========== #
    #n = 100
    #u_s_0 = parametrized_initial_u_and_times(u0, Lx, Ly, n)
    #time_array_x = p.zeros(n)
    #time_array_y = p.zeros(n)

    # =================== EXPERIMEnT InFO ==================== #
    txt_file =  open(r"surface_2layers_animations_and_info/{}_info.txt".format(file_name),"w+") 
    if experiment != "12":
      info = ["t_end: {} [s]\n".format(t_end),"\n",
              "lambda_0: {}\n".format(l_0),"\n",
              "lambda_1: {}\n".format(l_1),"\n",
              "rho_0: {} [kg/m^3]\n".format(r_0),"\n",
              "rho_1: {} [kg/m^3]\n".format(r_1),"\n",
              "mu_0: {}\n".format(m_0),"\n",
              "mu_1: {}\n".format(m_1),"\n",
              "depth of 1st layer: {} [m]".format(first_layer_depth),"\n"]
    else:
      info = ["Oblique layers",
              "t_end: {} [s]\n".format(t_end),"\n",
              "granite, marble, limestone1"]
              #"depth of layer: {} [m]".format(layers_depth),"\n"]
            
    txt_file.writelines(info)
    txt_file.close()


    rec_counter = 0

    t0 = time.time()
    while t < t_end - 0.5*dt:


        t += float(dt)

        if rank == 0:
            print('\n\rtime: {:.3f} (Progress: {:.2f}%)'.format(t, 100 * t / t_end),)
            print("time taken til this point: {}".format(time.time()-t0))
        # Update source term
        #f.t = t
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

        #  time_array_x, time_array_y = time_to_pml_border(u_s_0, u0, time_array_x,time_array_y, t, Lx, Ly, n)
        # Save solution to XDMF file format
        
        xdmf_file.write(u, t)
        if record and rec_counter % 2 == 0:

            pvd << (u0,t)
        rec_counter += 1

        #s = p.linspace(0,Lx+2*Ly, n)
        #file_time = open('filename_time_to_pml.obj', 'w') 
        #pickle.dump([s,time_array_x,time_array_y], file_time)



  #plt.plot(s, time_array_x)
  #plt.plot(s, time_array_y)
  #plt.show()
  # Close file

    xdmf_file.close()