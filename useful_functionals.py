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
import matplotlib.pyplot as plt
import os
from time_integration_and_compliance import N_dot1



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
    
