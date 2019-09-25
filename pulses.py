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

def modified_ricker_pulse(w_p, t):
    return ((0.25) * (t ** 2 - 0.5)* p.exp(-0.25 * t ** 2) - 13*p.exp(-13.5)) / (0.5 + 13*p.exp(13.5))

def ricker_wavelet(w_p, t):
    x = 0.5 *(w_p ** 2) * (t ** 2)
    return 0.001*(1 - x) * p.exp(- x)

def F_ricker_wavelet(w_p, w):
    c = 2 / p.sqrt(p.pi)

    return c * w ** 2 * p.exp(- (w / w_p) ** 2) / w_p ** 3

# ================= PULSE CLASSES  ================= #

class RickerPulse(UserExpression):
  def __init__(self, t, omega, amplitude, **kwargs):
    self.t     = t
    self.omega = omega
    self.amplitude = amplitude
    super().__init__(**kwargs)
  
  def eval(self, values, x):
    # Ricker pulse Parameters
    r  = ((x[0] - xc)**2 + (x[1] - yc)**2)**0.5
    
    u  = self.omega * self.t - 3*pow(6,0.5)
    Sp = (1.0 - (r / rd) ** 2) ** 3

    #if self.t <= 6 * pow(6, 0.5) / self.omega:

    Tp = ((0.25 * pow(u, 2) - 0.5) * exp(-0.25 * pow(u, 2)) \
         -13 * exp(-13.5)) / (0.5 + 13 * exp(-13.5))
    #else:
   #  Tp = 0

    values[0] = self.amplitude * physical_parameters.amplitude * Tp * Sp * (x[0] - xc) / r
    values[1] = self.amplitude * physical_parameters.amplitude * Tp * Sp * (x[1] - yc) / r

  def value_shape(self):
    return (2,)


class ModifiedRickerPulse(UserExpression):
  def __init__(self, t, omega, amplitude, center=0, **kwargs):
    super().__init__(**kwargs)
    self.t  = t
    self.omega  = omega * 2  * p.pi
    self.center = center
    self.amplitude = amplitude

  def eval(self, values, x):

    u  = self.omega*self.t - 3*pow(6,0.5)
    if abs(x[0] - self.center) < 0.25:
      Tp = ((0.25*pow(u,2) - 0.5)*exp(-0.25*pow(u,2)) - 13*exp(-13.5))/(0.5 + 13*exp(-13.5))
      values[0] = 0.0
      values[1] = self.amplitude*Tp   # [KPa]
    else:
      values[0] = 0.0
      values[1] = 0.0      #5000*Tp 


  def value_shape(self):
    return (2,)


class ClassicRickerPulse(UserExpression):
  def __init__(self, t, omega, amplitude, center=0, **kwargs):
    super().__init__(**kwargs)
    self.t  = t
    self.omega  = omega * 2 * p.pi # rads/s
    self.center = center
    self.amplitude = amplitude

  def eval(self, values, x):
    if abs(x[0] - self.center) < 0.25:
      values[0] = 0
      values[1] = self.amplitude * (1 - 0.5 * self.omega ** 2 * self.t ** 2) * exp( - 0.25 *  self.omega ** 2 * self.t ** 2) 
    else:
      values[0] = 0.0
      values[1] = 0.0      #5000*Tp 


  def value_shape(self):
    return (2,)


class Sine(UserExpression):
  def __init__(self, t, freq, amplitude, **kwargs):
      super().__init__(**kwargs)
      self.t  = t
      self.freq  =  2 * p.pi * freq # rads/s
      self.amplitude = amplitude

  def eval(values, x):
    values[0] = 0
    values[1] = self.amplitude * p.sin(self.omega_p * t)


  def value_shape(self):
    return (2,)


if __name__ == "__main__":
    w_p = 10

    T = p.arange(0, 200, 0.01)
    #plt.plot(T,ricker_wavelet(w_p, T))
    plt.plot(T,F_ricker_wavelet(100, T))

    plt.show()

