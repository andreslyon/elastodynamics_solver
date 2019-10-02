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
#from analysis_functions import *
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

class MultiLayer(UserExpression):
  def __init__(self, layer_properties, layers_depth, **kwargs):
      super().__init__(**kwargs)
      self.layer_properties = layer_properties
      self.layers_depth = layers_depth

  def eval(self, value, x):
    tol = 1e-14

    for i in range(1, len(self.layers_depth)):
 
      if abs(self.layers_depth[i - 1]) + tol <= abs(x[1]) and abs(x[1]) <= abs(self.layers_depth[i]) + tol:

          value[0] = self.layer_properties[i - 1]


class TwoLayeredProperties(UserExpression):
  def __init__(self, property_0, property_1, first_layer_depth, **kwargs):
    super().__init__(**kwargs)
    self.property_0 = property_0
    self.property_1 = property_1
    self.first_layer_depth = first_layer_depth

  def eval(self, value, x):
    tol = 1e-14
    if abs(x[1]) <= abs(self.first_layer_depth) + tol:
      value[0] = self.property_0
    else:
      value[0] = self.property_1
        

#  =========== HETEROGEnEOUS DOMAIn ========= #


class Layer(SubDomain):
  def __init__(self, Lx, Ly, Lpml, layer_start, layer_end, **kwargs):
    super().__init__(**kwargs)
    self.Lx = Lx
    self.Ly = Ly
    self.Lpml = Lpml
    self.layer_start = layer_start
    self.layer_end = layer_end
    self.tol = 10e-14

  def inside(self, x, on_boundary):
    return self.layer_start - self.tol <= abs(x[1]) <= self.layer_end + self.tol

class MaterialProperty(UserExpression):
  def __init__(self, prop_0, prop_1, subdomains,**kwargs):
    super().__init__(**kwargs)
    self.subdomains = subdomains
    self.prop_0 = prop_0
    self.prop_1 = prop_1
    #self._ufl_shape = ()

    #self._count = 0
  def eval_cell(self, values, x, cell):
    if self.subdomains[cell.index] == 0:
      values[0] = self.prop_0
    else:
      values[0] = self.prop_1
  def value_shape(self):
    return ()

  

# Update previous time step using Newmark scheme

# Dirichlet boundary
class Dirichlet(SubDomain):
  def __init__(self, Lx, Ly, Lpml, **kwargs):
    super().__init__(**kwargs)
    self.Lx = Lx
    self.Ly = Ly
    self.Lpml = Lpml
 

  def inside(self, x, on_boundary):
    return x[0] <= - 0.5 * self.Lx - self.Lpml + DOLFIN_EPS \
      or x[0] >= 0.5 * self.Lx + self.Lpml - DOLFIN_EPS \
      or x[1] <= - self.Ly - self.Lpml + DOLFIN_EPS and on_boundary


# Source domain
class Circle(SubDomain):
  def __init__(self, xc, yc, rd, **kwargs):
    super().__init__(**kwargs)
    self.xc = xc
    self.yc = yc
    self.rd = rd

  def inside(self, x, on_boundary):
    return pow(x[0] - self.xc, 2) + pow(x[1] - self.yc, 2) <= pow(self.rd, 2)



def mesh_generator(Lx, Ly, Lpml, nx, ny):
    return RectangleMesh(Point(- 0.5 * Lx - Lpml, 0.0), Point(0.5 * Lx + Lpml, - Ly - Lpml), nx, ny, "crossed")



def stable_dx(min_velocity, max_omega_p):
    wl_min = min_velocity / (1.636567 * max_omega_p)
    elements_per_wl = 8
    return wl_min / elements_per_wl

if __name__ == "__main__":
  pass