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
from time_integration_and_compliance import (N_dot, N_ddot, compliance, stable_dt, update, cfl_constant)
from domain_and_layers import (mesh_generator, stable_dx, ObliqueLayers, MultiLayer, TwoLayeredProperties, Dirichlet,
                               Circle, Layer, ObliqueLayer, MaterialProperty)
from materials import (Material, MaterialFromVelocities, read_materials, print_materials)
from user_interface import (soil_and_pulses_print, input_sources, save_info_oblique, type_of_medium_input)
from pml_functions import (alpha_0, alpha_1, alpha_2, beta_0, beta_1, beta_2)
from pulses import (ClassicRickerPulse, ModifiedRickerPulse, Cosine)
import numpy as p
import time
import pickle
from forward_problem import forward
from dolfin_adjoint import *


    
















