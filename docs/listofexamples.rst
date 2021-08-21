=====================
 例子的画廊
=====================

本页面包含 `source code
repository <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/>`_ 中的例子的概述。

例 1: Poisson 方程 with unit load
==========================================

This example solves the Poisson problem :math:`-\Delta u = 1` with the Dirichlet
boundary condition :math:`u = 0` in the unit square using piecewise-linear
triangular elements.

.. figure:: https://user-images.githubusercontent.com/973268/87638021-c3d1c280-c74b-11ea-9859-dd82555747f5.png

   The solution of Example 1.

See the `source code of Example 1 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex01.py>`_ for more information.
           
例 2: Kirchhoff 板弯曲问题
==========================================

这个例子求解双调和biharmonic Kirchhoff 板弯曲问题plate bending problem :math:`D
\Delta^2 u = f` 在单位正方形in the unit square with a constant loading :math:`f`, bending
stiffness :math:`D` and a combination of clamped, simply supported and free
boundary conditions.

.. figure:: https://user-images.githubusercontent.com/973268/87659951-f50bbc00-c766-11ea-8c0e-7de0e9e83714.png

   The solution of Example 2.

更多信息见 `source code of Example 2 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex02.py>`_ .

例 3: 线性弹性特征值问题
============================================

This example solves the linear elastic eigenvalue problem
:math:`\mathrm{div}\,\sigma(u)= \lambda u` with
the displacement fixed on the left hand side boundary.

.. figure:: https://user-images.githubusercontent.com/973268/87661134-cbec2b00-c768-11ea-81bc-f5455df7cc33.png

   例3的第五个本征模.

更多信息见  `source code of Example 3 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex03.py>`_ .

例 4: 线性化接触问题
=====================================

This example solves a single interation of the contact problem
between two elastic bodies using the Nitsche's method.
Triangular and quadrilateral second-order elements are used
in the discretization of the two elastic bodies.

.. figure:: https://user-images.githubusercontent.com/973268/87661313-1372b700-c769-11ea-89ee-db144986a25a.png

   The displaced meshes and the von Mises stress of Example 4.

更多信息见  `source code of Example 4 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex04.py>`_ .

例 7: 内部惩罚方法
==================================

This example solves the Poisson problem :math:`-\Delta u = 1` with :math:`u=0`
on the boundary using interior penalty discontinuous Galerkin method.
The finite element basis is piecewise-linear but discontinuous over
the element edges.

.. figure:: https://user-images.githubusercontent.com/973268/87662192-80d31780-c76a-11ea-9291-2d11920bc098.png

   The solution of Example 7.

更多信息见  `source code of Example 7 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex07.py>`_ .

例 8: Argyris 基函数
==================================

This example visualizes the :math:`C^1`-continuous fifth degree Argyris basis
functions on a simple triangular mesh.
This element can be used in the conforming discretization of biharmonic problems.

.. figure:: https://user-images.githubusercontent.com/973268/87662432-e0c9be00-c76a-11ea-85b9-711c6b34791e.png

   The Argyris basis functions of Example 8 corresponding to the middle node and
   the edges connected to it.

更多信息见  `source code of Example 8 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex08.py>`_ .

例 9: 三维 Poisson 方程
=============================================

This example solves :math:`-\Delta u = 1`
with :math:`u=0` on the boundary using tetrahedral elements and a preconditioned
conjugate gradient method.

.. note::

   This example will make use of the external packages `PyAMG
   <https://pypi.org/project/pyamg/>`__ or `pyamgcl
   <https://pypi.org/project/pyamgcl/>`__, if installed.

.. figure:: https://user-images.githubusercontent.com/973268/93183072-33abfb80-f743-11ea-9076-1324cbf28531.png

   The solution of Example 9 on a cross-section of the tetrahedral mesh.  The
   figure was created using `ParaView <https://www.paraview.org/>`__.

更多信息见  `source code of Example 9 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex09.py>`_ .

例 10: 非线性最小曲面问题
=============================================

This example solves the nonlinear minimal surface problem :math:`\nabla \cdot
\left(\frac{1}{\sqrt{1 + \|u\|^2}} \nabla u \right)= 0` with :math:`u=g`
prescribed on the boundary of the square domain.  The nonlinear problem is
linearized using the Newton's method with an analytical Jacobian calculated by
hand.

.. figure:: https://user-images.githubusercontent.com/973268/87663902-1c658780-c76d-11ea-9e00-324a18769ad2.png

   The solution of Example 10.

更多信息见  `source code of Example 10 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex10.py>`_ .

例 11: 三维线性弹性问题
===============================================

This example solves the three-dimensional linear elasticity equations
:math:`\mathrm{div}\,\sigma(u)=0` using trilinear hexahedral elements.
Dirichlet conditions are set on the opposing faces of a cube: one face remains
fixed and the other is displaced slightly outwards.

.. figure:: https://user-images.githubusercontent.com/973268/87685532-31054800-c78c-11ea-9b89-bc41dc0cb80c.png

   The displaced mesh of Example 11.  The figure was created using `ParaView
   <https://www.paraview.org/>`__.

更多信息见  `source code of Example 11 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex11.py>`_ .

例 12: 后处理
==============================================

This example demonstrates postprocessing the value of a functional, Boussinesq's k-factor.

.. figure:: https://user-images.githubusercontent.com/1588947/93292071-0127fe80-f828-11ea-8c9e-46590d280b69.png

   The solution of Example 12.

更多信息见  `source code of Example 12 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex12.py>`_ .

例 13: 带有混合边界条件的Laplace问题
==================================================

This example solves :math:`\Delta u = 0` in
:math:`\Omega=\{(x,y):1<x^2+y^2<4,~0<\theta<\pi/2\}`, where :math:`\tan \theta =
y/x`, with :math:`u = 0` on :math:`y = 0`, :math:`u = 1` on :math:`x =
0`, and :math:`\frac{\partial u}{\partial n} = 0` on the rest of the
boundary.

.. figure:: https://user-images.githubusercontent.com/1588947/94758580-d5d51000-03e0-11eb-8219-15cbba1d8c26.png

   The solution of Example 13.

更多信息见  `source code of Example 13 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex13.py>`_ .

.. _ex14:

例 14: 带有非齐次边界条件的Laplace问题
==========================================================

This example demonstrates how to impose coordinate-dependent Dirichlet
conditions for the Laplace equation :math:`\Delta u = 0`. The solution will
satisfy :math:`u=x^2 - y^2` on the boundary of the square domain.

.. figure:: https://user-images.githubusercontent.com/973268/87775119-3dda7800-c82e-11ea-8576-2219fcf31814.png

   The solution of Example 14.

更多信息见 `source code of Example 14 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex14.py>`_ .

例 15: 一维Poisson方程
============================================

This example solves :math:`-u'' = 1` in :math:`(0,1)` with the boundary
condition :math:`u(0)=u(1)=0`.

.. figure:: https://user-images.githubusercontent.com/973268/87775166-52b70b80-c82e-11ea-9009-c9fa0a9e28e8.png

   The solution of Example 15.

更多信息见  `source code of Example 15 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex15.py>`_ .

例 16: Legendre方程
===============================

This example solves the eigenvalue problem :math:`((1 - x^2) u')' + k u = 0` in
:math:`(-1,1)`.

.. figure:: https://user-images.githubusercontent.com/973268/87775206-65c9db80-c82e-11ea-8c49-bf191915602a.png

   The six first eigenmodes of Example 16.

更多信息见  `source code of Example 16 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex16.py>`_ .

例 17: 绝缘线
==========================

This example solves the steady heat conduction
with generation in an insulated wire. In radial
coordinates, the governing equations read: find :math:`T`
satisfying :math:`\nabla \cdot (k_0 \nabla T) + A = 0,~0<r<a`,
and
:math:`\nabla \cdot (k_1 \nabla T) = 0,~a<r<b`,
with the boundary condition
:math:`k_1 \frac{\partial T}{\partial r} + h T = 0` on :math:`r=b`.

.. figure:: https://user-images.githubusercontent.com/973268/87775309-8db93f00-c82e-11ea-9015-add2226ad01e.png

   The solution of Example 17.

更多信息见  `source code of Example 17 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex17.py>`_ .

例 18: Stokes方程
============================

This example solves for the creeping flow problem in the primitive variables,
i.e. velocity and pressure instead of the stream-function.  These are governed
by the Stokes momentum :math:`- \nu\Delta\boldsymbol{u} + \rho^{-1}\nabla p = \boldsymbol{f}` and the continuity equation :math:`\nabla\cdot\boldsymbol{u} = 0`.

.. figure:: https://user-images.githubusercontent.com/1588947/93292002-d6d64100-f827-11ea-9a0a-c64d5d2979b7.png

   The streamlines of Example 18.

更多信息见 `source code of Example 18 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex18.py>`_ .

例 19: 热传导方程
=========================

This example solves the heat equation :math:`\frac{\partial T}{\partial t} = \kappa\Delta T` in the domain :math:`|x|<w_0` and :math:`|y|<w_1` with the initial value :math:`T_0(x,y) = \cos\frac{\pi x}{2w_0}\cos\frac{\pi y}{2w_1}` using the generalized trapezoidal
rule ("theta method") and fast time-stepping by factorizing the evolution matrix once and for all.

.. figure:: https://user-images.githubusercontent.com/973268/87778846-7b420400-c834-11ea-8ff6-c439699b2802.gif

   The solution of Example 19.

更多信息见  `source code of Example 19 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex19.py>`_ .

例 20: Creeping flow via stream-function
=============================================

This example solves the creeping flow problem via the stream-function
formulation.
The stream-function :math:`\psi` for two-dimensional creeping flow is
governed by the biharmonic equation :math:`\nu \Delta^2\psi = \mathrm{rot}\,\boldsymbol{f}` where :math:`\nu` is the kinematic viscosity (assumed constant),
:math:`\boldsymbol{f}` the volumetric body-force, and :math:`\mathrm{rot}\,\boldsymbol{f} =
\partial f_y/\partial x - \partial f_x/\partial y`.  The boundary
conditions at a wall are that :math:`\psi` is constant (the wall is
impermeable) and that the normal component of its gradient vanishes (no
slip)

.. figure:: https://user-images.githubusercontent.com/1588947/93291998-d50c7d80-f827-11ea-861b-f24ed27072d0.png

   The velocity field of Example 20.

更多信息见  `source code of Example 20 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex20.py>`_ .

例 21: Structural vibration
================================

This example demonstrates the solution of a three-dimensional vector-valued
eigenvalue problem by considering the vibration of an elastic structure.

.. figure:: https://user-images.githubusercontent.com/973268/87779087-ebe92080-c834-11ea-9acc-d455b6124ad7.png

   An eigenmode本征模 of Example 21.

Structural vibration.

This example demonstrates the solution of a three-dimensional
vector-valued problem. For this purpose, we consider an elastic
eigenvalue problem.

The governing equation for the displacement of the elastic structure
:math:`\Omega` reads: find :math:`\boldsymbol{u} : \Omega \rightarrow
\mathbb{R}^3` satisfying

.. math::
   \rho \ddot{\boldsymbol{u}} = \mathrm{div}\,\boldsymbol{\sigma}(\boldsymbol{u}) + \rho \boldsymbol{g},

where :math:`\rho = 8050\,\frac{\mathrm{kg}}{\mathrm{m}^3}` is the
density, :math:`\boldsymbol{g}` is the gravitational acceleration and
:math:`\boldsymbol{\sigma}` is the linear elastic stress tensor
defined via

.. math::
   \begin{aligned}
   \boldsymbol{\sigma}(\boldsymbol{w}) &= 2 \mu \boldsymbol{\epsilon}(\boldsymbol{w}) + \lambda \mathrm{tr}\,\boldsymbol{\epsilon}(\boldsymbol{w}) \boldsymbol{I}, \\
   \boldsymbol{\epsilon}(\boldsymbol{w}) &= \frac12( \nabla \boldsymbol{w} + \nabla \boldsymbol{w}^T).
   \end{aligned}

Moreover, the Lamé parameters are given by

.. math::
   \lambda = \frac{E}{2(1 + \nu)}, \quad \mu = \frac{E \nu}{(1+ \nu)(1 - 2 \nu)},

where the Young's modulus :math:`E=200\cdot 10^9\,\text{Pa}`
and the Poisson ratio :math:`\nu = 0.3`.

We consider two kinds of boundary conditions. On a *fixed part* of the boundary, :math:`\Gamma_D \subset \partial \Omega`, the displacement field :math:`\boldsymbol{u}` satisfies

.. math::
   \boldsymbol{u}|_{\Gamma_D} = \boldsymbol{0}.

Moreover, on a *free part* of the boundary, :math:`\Gamma_N = \partial \Omega \setminus \Gamma_D`, the *traction vector* :math:`\boldsymbol{\sigma}(\boldsymbol{u})\boldsymbol{n}` satisfies

.. math::
   \boldsymbol{\sigma}(\boldsymbol{u})\boldsymbol{n} \cdot \boldsymbol{n}|_{\Gamma_N} = 0,

where :math:`\boldsymbol{n}` denotes the outward normal.

Neglecting the gravitational acceleration :math:`\boldsymbol{g}` and
assuming a periodic solution of the form

.. math::
   \boldsymbol{u}(\boldsymbol{x},t) = \boldsymbol{w}(\boldsymbol{x}) \sin \omega t,

leads to the following eigenvalue problem with :math:`\boldsymbol{w}` and :math:`\omega` as unknowns:

.. math::
   \mathrm{div}\,\boldsymbol{\sigma}(\boldsymbol{w}) = \rho \omega^2 \boldsymbol{w}.

The weak formulation of the problem reads: find :math:`(\boldsymbol{w},\omega) \in V \times \mathbb{R}` satisfying

.. math::
   (\boldsymbol{\sigma}(\boldsymbol{w}), \boldsymbol{\epsilon}(\boldsymbol{v})) = \rho \omega^2 (\boldsymbol{w}, \boldsymbol{v}) \quad \forall \boldsymbol{v} \in V,

where the variational space :math:`V` is defined as

.. math::
   V = \{ \boldsymbol{w} \in [H^1(\Omega)]^3 : \boldsymbol{w}|_{\Gamma_D} = \boldsymbol{0} \}.

The bilinear form for the problem can be found from
:func:`skfem.models.elasticity.linear_elasticity`.  Moreover, the mesh
for the problem is loaded from an external file *beams.msh*, which is
included in the source code distribution.

更多信息见  `source code of Example 21 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex21.py>`_ .

例 22: 自适应Poisson方程
=====================================

This example solves Example 1 adaptively in an L-shaped domain.
Using linear elements, the error indicators read :math:`\eta_K^2 = h_K^2 \|f\|_{0,K}^2` and :math:`\eta_E^2 = h_E \| [[\nabla u_h \cdot n ]] \|_{0,E}^2`   
for each element :math:`K` and
edge :math:`E`.

.. figure:: https://user-images.githubusercontent.com/973268/87779195-15a24780-c835-11ea-9a18-767092ae9467.png

   例22的自适应加密网格.

Adaptive Poisson equation.

This example solves `ex01.py` adaptively in an L-shaped domain.
Using linear elements, the error indicators read

.. math::
   \eta_K^2 = h_K^2 \|f\|_{0,K}^2

for each element :math:`K`, and

.. math::
   \eta_E^2 = h_E \| [[\nabla u_h \cdot n ]] \|_{0,E}^2

for each edge :math:`E`.

更多信息见  `source code of Example 22 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex22.py>`_ .

例 23: Bratu-Gelfand
=========================

This example solves the Bratu-Gelfand two-point boundary value problem :math:`u'' + \lambda \mathrm e^u = 0`, :math:`0 < x < 1`,
with :math:`u(0)=u(1)=0` and where :math:`\lambda > 0` is a parameter.

.. note::
   This example requires the external package `pacopy 0.1.2 <https://pypi.org/project/pacopy/0.1.2>`__.

.. figure:: https://user-images.githubusercontent.com/973268/87779278-38ccf700-c835-11ea-955a-b77a0336b791.png

   The results of Example 23.


更多信息见 `source code of Example 23 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex23.py>`_ .

例 24: Stokes flow with inhomogeneous boundary conditions
==============================================================

This example solves the Stokes flow over a backward-facing step
with a parabolic velocity profile at the inlet.

.. figure:: https://user-images.githubusercontent.com/973268/87858848-92b6e500-c939-11ea-81f9-cc51f254d19e.png

   The streamlines of Example 24.

更多信息见  `source code of Example 24 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex24.py>`_ .

例 25: Forced convection
=============================

This example solves the plane Graetz problem with the governing
advection-diffusion equation :math:`\mathrm{Pe} \;u\frac{\partial T}{\partial x}
= \nabla^2 T` where the velocity profile is :math:`u (y) = 6 y (1 - y)` and the
Péclet number :math:`\mathrm{Pe}` is the mean velocity times the width divided
by the thermal diffusivity.

.. figure:: https://user-images.githubusercontent.com/973268/87858907-f8a36c80-c939-11ea-87a2-7357d5f073b1.png

   The solution of Example 25.

更多信息见  `source code of Example 25 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex25.py>`_ .

例 26: Restricting problem to a subdomain
==============================================

This example extends Example 17 by restricting the solution to a subdomain.

.. figure:: https://user-images.githubusercontent.com/973268/87858933-3902ea80-c93a-11ea-9d54-464235ab6325.png

   The solution of Example 26.

更多信息见  `source code of Example 26 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex26.py>`_ .

例 27: Backward-facing step
================================

This example uses `pacopy 0.1.2 <https://pypi.org/project/pacopy/0.1.2>`__ to extend
the Stokes equations over a backward-facing step (Example 24) to finite Reynolds
number; this means defining a residual for the nonlinear problem and its
derivatives with respect to the solution and to the Reynolds number.

.. note::
   This example requires the external package `pacopy 0.1.2 <https://pypi.org/project/pacopy/0.1.2>`__.

.. figure:: https://user-images.githubusercontent.com/973268/87858972-97c86400-c93a-11ea-86e4-66f870b03e48.png

   The streamlines of Example 27 for :math:`\mathrm{Re}=750`.

更多信息见  `source code of Example 27 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex27.py>`_ .

例 28: Conjugate heat transfer
===================================

This example extends Example 25 to conjugate heat transfer by giving a finite
thickness and thermal conductivity to one of the walls.  The example is modified
to a configuration for which there exists a fully developed solution which can be
found in closed form: given a uniform heat flux over each of the walls, the
temperature field asymptotically is the superposition of a uniform longitudinal
gradient and a transverse profile.

.. note::
   This example requires the external package
   `pygmsh <https://pypi.org/project/pygmsh/>`__.

.. figure:: https://user-images.githubusercontent.com/973268/87859005-c0505e00-c93a-11ea-9a78-72603edc242a.png

   The solution of Example 28.

更多信息见  `source code of Example 28 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex28.py>`_ .

例 29: Linear hydrodynamic stability
=========================================

The linear stability of one-dimensional solutions of the Navier-Stokes equations
is governed by the `Orr-Sommerfeld equation <https://en.wikipedia.org/wiki/Orr%E2%80%93Sommerfeld_equation>`_.  This is expressed in terms of the stream-function
:math:`\phi` of the perturbation, giving a two-point boundary value problem      
:math:`\alpha\phi(\pm 1) = \phi'(\pm 1) = 0`
for a complex fourth-order ordinary differential equation,

.. math::
   \left(\alpha^2-\frac{\mathrm d^2}{\mathrm dz^2}\right)^2\phi
   = (\mathrm j\alpha R)\left\{
     (c - U)\left(\alpha^2-\frac{\mathrm d^2}{\mathrm dz^2}\right)\phi
     - U''\phi,
   \right\}
   
where :math:`U(z)` is the base velocity profile, :math:`c` and :math:`\alpha`
are the wavespeed and wavenumber of the disturbance, and :math:`R` is the
Reynolds number.

.. figure:: https://user-images.githubusercontent.com/973268/87859022-e0801d00-c93a-11ea-978f-b1930627010b.png

   The results of Example 29.

更多信息见  `source code of Example 29 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex29.py>`_ .

例 30: Krylov-Uzawa method for the Stokes equation
=======================================================

This example solves the Stokes equation iteratively in a square domain.

.. figure:: https://user-images.githubusercontent.com/973268/87859044-06a5bd00-c93b-11ea-84c2-9fbb9fc6e832.png

   The pressure field of Example 30.

更多信息见  `source code of Example 30 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex30.py>`_ .

例 31: 曲边元
===========================

This example solves the eigenvalue problem :math:`-\Delta u = \lambda u`
with the boundary condition :math:`u|_{\partial \Omega} = 0` using isoparametric
mapping via biquadratic basis and finite element approximation using fifth-order
quadrilaterals.

.. figure:: https://user-images.githubusercontent.com/973268/87859068-32c13e00-c93b-11ea-984d-684e1e4c5066.png

   An eigenmode of Example 31 in a curved mesh.

更多信息见  `source code of Example 31 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex31.py>`_ .

例 32: 块对角预处理Stokes求解器
=========================================================

This example solves the Stokes problem in three dimensions, with an
algorithm that scales to reasonably fine meshes (a million tetrahedra in a few
minutes).

.. note::
   This examples requires an implementation of algebraic multigrid (either `pyamgcl <https://pypi.org/project/pyamgcl>`_ or `pyamg <https://pypi.org/project/pyamg/>`_).

.. figure:: https://user-images.githubusercontent.com/1588947/96520786-8a18d680-12bb-11eb-981a-c3388f2c8e35.png

   The velocity and pressure fields of Example 32, clipped in the plane of spanwise symmetry, *z* = 0.
   The figure was created using `ParaView <https://www.paraview.org/>`_ 5.8.1.

更多信息见  `source code of Example 32 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex32.py>`_ .

例 33: H(curl) 协调模型问题
============================================


This example solves the vector-valued problem :math:`\nabla \times \nabla \times
E + E = f` in domain :math:`\Omega = [-1, 1]^3` with the boundary condition
:math:`E \times n|_{\partial \Omega} = 0` using the lowest order Nédélec edge
element.

.. figure:: https://user-images.githubusercontent.com/973268/87859239-47520600-c93c-11ea-8241-d62fdfd2a9a2.png

   The solution of Example 33 with the colors given by the magnitude
   of the vector field.
   The figure was created using `ParaView <https://www.paraview.org/>`__.

更多信息见  `source code of Example 33 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex33.py>`_ .

例 34: Euler-Bernoulli beam
================================


This example solves the Euler-Bernoulli beam equation
:math:`(EI u'')'' = 1`
with the boundary conditions
:math:`u(0)=u'(0) = 0` and using cubic Hermite elements.
The exact solution at :math:`x=1` is :math:`u(1)=1/8`.

.. figure:: https://user-images.githubusercontent.com/973268/87859267-749eb400-c93c-11ea-82cd-2d488fda39d4.png

   The solution of Example 34.

更多信息见  `source code of Example 34 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex34.py>`_ .

例 35: Characteristic impedance and velocity factor
========================================================

This example solves the series inductance (per meter) and parallel capacitance
(per meter) of RG316 coaxial cable. These values are then used to compute the
characteristic impedance and velocity factor of the cable.

.. figure:: https://user-images.githubusercontent.com/973268/87859275-85e7c080-c93c-11ea-9e62-3a9a8ee86070.png

   The results of Example 35.

更多信息见  `source code of Example 35 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex35.py>`_ .

例 36: Nearly incompressible hyperelasticity
=================================================

This example demonstrates the implementation of a two field mixed formulation
for nearly incompressible Neo-Hookean solids.

.. figure:: https://user-images.githubusercontent.com/22624037/91212007-4055aa80-e6d5-11ea-8572-f27986887331.png

   The displacement contour of Example 36.
   The figure was created using `ParaView <https://www.paraview.org/>`__.

See the `source code of Example 36 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex36.py>`_ for more information.

例题 37: 混合Poisson方程
==================================

This example solves the mixed formulation of the Poisson equation
using the lowest order Raviart-Thomas elements.

.. figure:: https://user-images.githubusercontent.com/973268/93132097-c2862d00-f6dd-11ea-97ad-40aaf2732ad1.png

   The piecewise constant solution field.
   The figure was created using `ParaView <https://www.paraview.org/>`__.

更多信息见  `source code of Example 37 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex37.py>`_ .

例 38: Point source
========================

Point sources require different assembly to other linear forms.

This example computes the Green's function for a disk; i.e. the solution of
the Dirichlet problem for the Poisson equation with the source term
concentrated at a single interior point, :math:`\Delta u = \delta (\mathbf x - \mathbf s)`.

.. figure:: https://user-images.githubusercontent.com/1588947/115502511-5cd3d200-a2b8-11eb-9929-92ed9550ced8.png

    The scalar potential in the disk with point source at (0.3, 0.2).

更多信息见  `source code of Example 38 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex38.py>`_ .

例 39: 一维热传导方程
=========================================

This examples reduces the two-dimensional heat equation of Example 19 to
demonstrate the special post-processing required.

.. figure:: https://user-images.githubusercontent.com/1588947/127958860-6454e542-67ba-4e94-8053-5175da201daa.gif

   The solution of Example 39.

更多信息见 `source code of Example 39 <https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex39.py>`_ .
