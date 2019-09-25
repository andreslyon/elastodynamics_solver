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

def stable_dt(dx, c_p):
    return dx / c_p
