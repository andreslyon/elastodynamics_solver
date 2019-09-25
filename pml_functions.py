import os
import sys
from dolfin import (MPI, SubDomain, UserExpression, VectorElement, TensorElement,
                    MixedElement, FiniteElement, Function, FunctionSpace,
                    RectangleMesh, MeshFunction, TestFunctions, TrialFunctions,
                    DirichletBC, parameters, set_log_level, LogLevel, Point,
                    DOLFIN_EPS, DOLFIN_PI, interpolate, Constant, KrylovSolver,
                    XDMFFile, File, plot)
from ufl import (as_tensor, Measure, lhs, rhs, inner, grad, exp, tr, Identity)
import numpy as p
import time


def beta_0(m, V_r, R, Lpml):
    return (m + 1) * V_r * p.log( 1 / abs(R)) / (2 * Lpml)

def alpha_0(m, car_dim, R, Lpml):
    return (m + 1) * car_dim *  p.log(1 / abs(R)) / (2 * Lpml)

class alpha_1(UserExpression):
  def __init__(self, alpha_0, Lx, Lpml, **kwargs):
    super().__init__(**kwargs)
    self.alpha_0 = alpha_0
    self.Lx = Lx
    self.Lpml = Lpml

  def eval(self, values, x, **kwargs):
    values[0] = 1.0 + self.alpha_0 * pow((abs(x[0]) - 0.5 * self.Lx + abs(abs(x[0]) - 0.5 * self.Lx )) / (2 * self.Lpml), 2)


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
        values[0] = self.beta_0 * pow((abs(x[0]) - 0.5 * self.Lx + abs(abs(x[0]) - 0.5 * self.Lx))/(2 * self.Lpml), 2)


class beta_2(UserExpression):
    def __init__(self, beta_0, Ly, Lpml, **kwargs):
        super().__init__(**kwargs)
        self.beta_0 = beta_0
        self.Ly = Ly
        self.Lpml = Lpml  
    def eval(self, values, x, **kwargs):
        values[0] = self.beta_0 * pow((-(x[1] + self.Ly) + abs(x[1] + self.Ly))/(2 * self.Lpml), 2)
