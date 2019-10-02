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
import time

def N_ddot(arg, u0, a0, v0, dt, beta):
    """Acceleration """
    output = (arg - u0 - dt * v0) / (beta * dt * dt) - a0 * (1 - 2 * beta)/(2 * beta)
    return output



def N_dot(arg, u0, v0, a0, dt, beta, gamma):
    return gamma * (arg - u0) / (beta * dt) - v0 * (gamma - beta) / beta - a0 * dt * (gamma - 2 * beta)/(2 * beta)


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

def compliance(sigma, u, mu, lmbda):
    return sigma / (2 * mu) - lmbda / (4 * mu * (lmbda + mu)) * tr(sigma) * Identity(u.geometric_dimension())

class Compliance(UserExpression):
    def __init__(self, sigma, u, mu, lmbda, subdomains, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.sigma = sigma
        self.u = u
        self.mu = mu
        self.lmbda = lmbda
        self._ufl_shape = (2,2)
        #self._count = 0
    def eval_cell(self, values, x, cell):
        m = self.mu.eval_cell(x, cell)
        l = self.lmbda.evall_cell(x, cell)

        C = self.sigma / (2 *  m) -  l / (4 *  m * (l + m)) * tr(self.sigma) * Identity(self.u.geometric_dimension())
        print(C)

        #C = self.sigma / (2 *  self.mu) -  self.lmbda / (4 *  self.mu *
         #( self.lmbda +  self.mu)) * tr( self.sigma) * Identity( self.u.geometric_dimension())
        print(type(values))
        print(type(C))

        time.sleep(10)

        #print(C[0,0])
        #time.sleep(10)
        
        ##values[0][0] = C[0, 0]
        #values[0][1] = C[0, 1]
        #values[1][0] = C[1, 0]
        #values[1][1] = C[1, 1]

    def value_shape(self):
        return (2, 2)  

def stable_dt(dx, c_p):
    cfl_constant = 0.3125
    return cfl_constant * dx / c_p

def cfl_constant(v, dt, dx):
    return dt * v / dx

if __name__ == "__main__":
    a = p.array([[1,2],[3,4]])
    print(a)
    print(a[0][0])