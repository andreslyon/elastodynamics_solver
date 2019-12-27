import os
import sys
from dolfin import (MPI, SubDomain, UserExpression, VectorElement, TensorElement,
                    MixedElement, FiniteElement, Function, FunctionSpace,
                    RectangleMesh, MeshFunction, TestFunctions, TrialFunctions,
                    DirichletBC, parameters, set_log_level, LogLevel, Point,
                    DOLFIN_EPS, DOLFIN_PI, interpolate, Constant, KrylovSolver,
                    XDMFFile, File, plot)
from dolfin.fem.assembling import assemble
from ufl import (as_tensor, Measure, lhs, rhs, inner, grad, exp, tr, Identity, sqrt)
import numpy as p
import matplotlib.pyplot as plt
import os
from time_integration_and_compliance import N_dot

from materials import cp, cs

def L2_norm(u, dx):    
    energy = inner(u, u) * dx
    E = assemble(energy)
    return p.sqrt(E)

def tn_reg(lmda, mu, dx, R_lmda=1, R_mu=1):
    grad_lamda = grad(lmda)
    grad_mu = grad(mu)
    reg = 0.5 * R_lmda * inner(grad_lamda, grad_lamda) * dx + \
        0.5 * R_mu * inner(grad_mu, grad_mu) * dx
    Reg = assemble(reg)
    return Reg
    
def tv_reg(lmda, mu, dx, R_lmda=1, R_mu=1, eps=1e-10):



    grad_lamda = grad(lmda)
    grad_mu = grad(mu)
    integrand_lmda = R_lmda * sqrt(inner(grad_lamda, grad_lamda) + eps) * dx
    integrand_mu = R_mu * sqrt(inner(grad_mu, grad_mu) + eps) * dx
    return assemble(integrand_lmda + integrand_mu)


def misfit_functional():
    pass

if __name__ == "__main__":
    m = p.array([74263.42, 74263.42, 74263.42, 74263.42, 74263.42, 74263.42])
    l = p.array([49508.94666666666, 990178.9333333332, 49508.94666666666, 990178.9333333332, 49508.94666666666, 495089.4666666666])
    rho = 2200
    print(cp(m,l,rho))