=============
How-to 手册
=============

本节包含scikit-fem特色的面向目标的手册.

选择有限元
=========================

对于选择一个 :class:`~skfem.element.Element` 类这里有一些用法说明.  
首先, 单元类型的命名反映它们与网格类型的兼容性:

>>> from skfem.element import ElementTriP1
>>> ElementTriP1.refdom
<class 'skfem.refdom.RefTri'>

其次, 选择的有限元应该与逼近的PDE兼容.  这里有一般准则

* 使用 :class:`~skfem.element.ElementH1` 子类, e.g. 对标准二阶问题,
  :class:`~skfem.element.ElementTriP1`, :class:`~skfem.element.ElementTriP2`,
  :class:`~skfem.element.ElementQuad2`, :class:`~skfem.element.ElementTetP2` 或 
  :class:`~skfem.element.ElementHex2` .
* 离散向量问题要么通过手动建立带有标量有限元的块矩阵
  (e.g., 使用 ``scipy.sparse.bmat``) , 要么通过使用
  :class:`~skfem.element.ElementVector` 或
  :class:`~skfem.element.ElementComposite` 这是对块矩阵创建的抽象. 
* 特别注意约束问题, e.g., the Stokes方程组，它可能需要特殊有限元的使用
  :class:`~skfem.element.ElementTriMini`.
* Use subclasses of :class:`~skfem.element.ElementHdiv` 或
  :class:`~skfem.element.ElementHcurl`, e.g.,
  :class:`~skfem.element.ElementTriRT0` 或 :class:`~skfem.element.ElementTetN0`,
  对于具有较少正则解的混合问题.
* 使用 :class:`ElementGlobal` 子类, e.g. 对于四阶问题, 或 如果有后处理要求, e.g., 对高阶导数的需求 :class:`ElementTriMorley` 或 :class:`ElementTriArgyris`.

第三, 有限元应使用自由度，这自由度与要施加的基本边界条件相关。更多信息见 :ref:`finddofs` .

有限元列表见 :mod:`skfem.element` 文档.


.. _predefined:

在形式中使用离散函数
=================================

经常在形式定义中使用一个前一个解向量, e.g., 当求解非线性问题时或者当计算泛函时。
一个简单的固定点迭代

.. math::

   \begin{aligned}
      -\nabla \cdot ((u + 1)\nabla u) &= 1 \quad \text{in $\Omega$}, \\
      u &= 0 \quad \text{on $\partial \Omega$},
   \end{aligned}

corresponds to repeatedly
找 :math:`u_{k+1} \in H^1_0(\Omega)` 满足

.. math::

   \int_\Omega (u_{k} + 1) \nabla u_{k+1} \cdot \nabla v \,\mathrm{d}x = \int_\Omega v\,\mathrm{d}x

对每一个 :math:`v \in H^1_0(\Omega)`。

参数 ``w`` 用于定义这样的形式:

.. doctest::

   >>> import skfem as fem
   >>> from skfem.models.poisson import unit_load
   >>> from skfem.helpers import grad, dot
   >>> @fem.BilinearForm
   ... def bilinf(u, v, w):
   ...     return (w.u_k + 1.) * dot(grad(u), grad(v))

前一个解 :math:`u_k` 作为一个关键参数，在
:meth:`~skfem.assembly.BilinearForm.assemble` 使用到:

.. doctest::

   >>> m = fem.MeshTri().refined(3)
   >>> basis = fem.Basis(m, fem.ElementTriP1())
   >>> b = unit_load.assemble(basis)
   >>> x = 0. * b.copy()
   >>> for itr in range(10):  # fixed point iteration
   ...     A = bilinf.assemble(basis, u_k=basis.interpolate(x))
   ...     x = fem.solve(*fem.condense(A, b, I=m.interior_nodes()))
   ...     print(x.max())
   0.07278262867647059
   0.07030433694174187
   0.07036045457157739
   0.07035940302769318
   0.07035942072395032
   0.07035942044353624
   0.07035942044783286
   0.07035942044776827
   0.07035942044776916
   0.07035942044776922

形式定义之中, ``w`` 是用户提供的参数和附加默认值的字典。
默认情况下, ``w['x']`` (也作 ``w.x`` 访问) 对应于总体坐标和 ``w['h']`` (也作 ``w.h`` 访问) 对应于局部网格参数。


.. _finddofs:

找自由度
==========================

通常目标是限制自由度在边界的某一部分。 
目前找自由度的主要工具是
:meth:`skfem.assembly.Basis.find_dofs` 和
:meth:`skfem.assembly.Basis.get_dofs`.  我们之后用例子来演示。

.. doctest::

   >>> from skfem import MeshTri, Basis, ElementTriP2
   >>> m = MeshTri().refined(2)
   >>> basis = Basis(m, ElementTriP2())

我们首先属于左面边界的界面集合。

.. doctest::

   >>> m.facets_satisfying(lambda x: x[0] == 0.)
   array([ 1,  5, 14, 15])

接下来我们提供界面指示到
:meth:`skfem.assembly.Basis.get_dofs`

.. doctest::

   >>> dofs = basis.get_dofs(m.facets_satisfying(lambda x: x[0] == 0.))
   >>> dofs.nodal
   {'u': array([ 0,  2,  5, 10, 14])}
   >>> dofs.facet
   {'u': array([26, 30, 39, 40])}

上面字典中的值根据下表表示自由度的类型:

+-----------+---------------------------------------------------------------+
| Key       | Description                                                   |
+===========+===============================================================+
| ``u``     | Point value                                                   |
+-----------+---------------------------------------------------------------+
| ``u_n``   | Normal derivative                                             |
+-----------+---------------------------------------------------------------+
| ``u_x``   | Partial derivative w.r.t. :math:`x`                           |
+-----------+---------------------------------------------------------------+
| ``u_xx``  | Second partial derivative w.r.t :math:`x`                     |
+-----------+---------------------------------------------------------------+
| ``u^n``   | Normal component of a vector field (e.g. Raviart-Thomas)      |
+-----------+---------------------------------------------------------------+
| ``u^t``   | Tangential component of a vector field (e.g. Nédélec)         |
+-----------+---------------------------------------------------------------+
| ``u^1``   | First component of a vector field                             |
+-----------+---------------------------------------------------------------+
| ``u^1_x`` | Partial derivative of the first component w.r.t. :math:`x`    |
+-----------+---------------------------------------------------------------+
| ``u^1^1`` | First component of the first component in a composite field   |
+-----------+---------------------------------------------------------------+
| ``NA``    | Description not available (e.g. hierarchical or bubble DOF's) |
+-----------+---------------------------------------------------------------+

所有自由度列表(属于左面边界belonging to the left boundary) 可以通过如下方式获取:

.. doctest::

   >>> dofs.flatten()
   array([ 0,  2,  5, 10, 14, 26, 30, 39, 40])
   
很多自由度类型与指定总体坐标相关。这些所谓自由度位置可以按如下方式找到:

.. doctest::

   >>> basis.doflocs[:, dofs.flatten()]
   array([[0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
          [0.   , 1.   , 0.5  , 0.25 , 0.75 , 0.125, 0.875, 0.375, 0.625]])

更多详细信息见 :ref:`dofindexing` 。

通过投影创建离散函数
==========================================

定义边界自由度的值并不总那么简单, e.g., 当自由度不表示一个点处的函数或者另外别的直观的量。

那么实现通过求解满足下述方程的函数
:math:`\widetilde{u_0} \in V_h` 将边界数据 :math:`u_0` 的 :math:`L^2` 投影映到有限元空间 :math:`V_h` ，这是可行的，

.. math::

   \int_{\partial \Omega} \widetilde{u_0} v\,\mathrm{d}s = \int_{\partial \Omega} u_0 v\,\mathrm{d}s\quad \forall v \in V_h,

并且在区域内所有自由度均为零。

下面我们显现地求解上面的变分问题:

.. doctest::

   >>> import skfem as fem
   >>> m = fem.MeshQuad()
   >>> basis = fem.FacetBasis(m, fem.ElementQuadP(3))
   >>> u_0 = lambda x: (x[0] * x[1]) ** 3
   >>> M = fem.BilinearForm(lambda u, v, w: u * v).assemble(basis)
   >>> f = fem.LinearForm(lambda v, w: u_0(w.x) * v).assemble(basis)
   >>> x = fem.solve(*fem.condense(M, f, I=basis.get_dofs()))
   >>> x
   array([ 2.87802132e-16,  1.62145397e-16,  1.00000000e+00,  1.66533454e-16,
           4.59225774e-16, -4.41713127e-16,  4.63704316e-16,  1.25333771e-16,
           6.12372436e-01,  1.58113883e-01,  6.12372436e-01,  1.58113883e-01,
           0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])

或者, 可使用 :func:`skfem.utils.projection` ，可以实现同样的功能:

.. doctest::

   >>> fem.projection(u_0, basis, I=basis.get_dofs(), expand=True)
   array([ 2.87802132e-16,  1.62145397e-16,  1.00000000e+00,  1.66533454e-16,
           4.59225774e-16, -4.41713127e-16,  4.63704316e-16,  1.25333771e-16,
           6.12372436e-01,  1.58113883e-01,  6.12372436e-01,  1.58113883e-01,
           0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
