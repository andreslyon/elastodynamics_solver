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

def modified_ricker_pulse(w_p, t):
    return p.piecewise(t, [ t <= 6*p.sqrt(6)/w_p, t > 6*p.sqrt(6)/w_p], [lambda t:  
      ((0.25) * ((w_p*t -3 * p.sqrt(6)) ** 2 - 0.5)* p.exp(-0.25 * (w_p*t -3 * p.sqrt(6)) ** 2) - 13*p.exp(-13.5)) / (0.5 + 13*p.exp(13.5)), lambda t: 0])
def ricker_wavelet(w_p, t):
    x = 0.5 *(w_p ** 2) * (t ** 2)
    return 0.001*(1 - x) * p.exp(- x)

def F_ricker_wavelet(w_p, w):
    c = 2 / p.sqrt(p.pi)

    return c * w ** 2 * p.exp(- (w / w_p) ** 2) / w_p ** 3

# ================= PULSE CLASSES  ================= #


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
  
  def pulse_info(self):
    info = "Pulse Type: Modified Ricker Pulse\n" + "Peak frequency [rad/s]: {}\n".format(self.omega) + "Amplitude [_]: {}\n".format(self.amplitude) + "center [m]: {}\n".format(self.center)
    return info




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

  def pulse_info(self):
    info = "Pulse Type: Classic Ricker Pulse\n" + "Peak frequency [rad/s]: {}\n".format(self.omega) + "Amplitude [_]: {}\n".format(self.amplitude) + "center [m]: {}\n".format(self.center)
    return info


class Cosine(UserExpression):
  def __init__(self, t, omega, amplitude, center=0, **kwargs):
      super().__init__(**kwargs)
      self.t  = t
      self.omega  =  2 * p.pi * omega # rads/s
      self.amplitude = amplitude
      self.center = center

  def eval(self, values, x):
    if abs(x[0] - self.center) < 0.25:
      values[0] = 0
      values[1] = self.amplitude * p.cos(self.omega * self.t)
    else:
      values[0] = 0
      values[1] = 0


  def value_shape(self):
    return (2,)

  def pulse_info(self):
    info = "Pulse Type: Cosine\n" + "Peak frequency [rad/s]: {}\n".format(self.omega) + "Amplitude [_]: {}\n".format(self.amplitude) + "center [m]: {}\n".format(self.center)
    return info


if __name__ == "__main__":
    w_0 = 2 * p.pi * 10
    w_1 = 2 * p.pi * 1

    T = p.arange(0, 1, 0.01)

    plt.plot(T,modified_ricker_pulse(w_0, T),color="red")
    plt.plot(T,modified_ricker_pulse(w_1, T),color="blue")

    plt.show()

