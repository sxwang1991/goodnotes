.. _gettingstarted:

=================
 新手入门
=================

本入门指南为你介绍scikit-fem。假设你已经掌握了有限元方法的基础。

第零步: 安装 scikit-fem
==========================

如果你的电脑支持Python安装, 你可以运行

.. code-block:: bash

   pip3 install scikit-fem

你也可以在浏览器中尝试 `Google Colab <https://colab.research.google.com/>`_ ，运行如下代码安装 scikit-fem

.. code-block:: bash

   !pip install scikit-fem

第一步: 阐明问题
===========================

入门指南中求解Poisson方程

.. math::
   \begin{aligned}
        -\Delta u &= f \quad && \text{in $\Omega$,} \\
        u &= 0 \quad && \text{on $\partial \Omega$,}
   \end{aligned}

这里 :math:`\Omega = (0, 1)^2` 方形区域
和 :math:`f(x,y)=\sin \pi x \sin \pi y`.

弱形式为:
找 :math:`u \in V` 满足

.. math::
   \int_\Omega \nabla u \cdot \nabla v \,\mathrm{d}x = \int_\Omega fv\,\mathrm{d}x \quad \forall v \in V,

这里 :math:`V = H^1_0(\Omega)`.

第二步: 表成代码
=================================

接下来我们写出形式

.. math::

   a(u, v) = \int_\Omega \nabla u \cdot \nabla v \,\mathrm{d}x \quad \text{and} \quad L(v) = \int_\Omega f v \,\mathrm{d}x

作为源代码.  每个形式写成一个函数，代码如下

.. doctest::

   >>> import skfem as fem
   >>> from skfem.helpers import dot, grad  # helpers使形式更美
   >>> @fem.BilinearForm
   ... def a(u, v, _):
   ...     return dot(grad(u), grad(v))

.. doctest::

   >>> import numpy as np
   >>> @fem.LinearForm
   ... def L(v, w):
   ...     x, y = w.x  # 总体坐标
   ...     f = np.sin(np.pi * x) * np.sin(np.pi * y)
   ...     return f * v

第三步: 创建网格
=====================

网格类默认结构体 :class:`~skfem.mesh.Mesh` 初始化单位正方形：

.. doctest::

   >>> mesh = fem.MeshTri().refined(3)  # 加密三次
   >>> mesh
   Triangular mesh with 81 vertices and 128 elements.


第四步: 定义基函数
======================

网格连同有限元形成总体基函数.
这里我们选择分片线性基函数:

.. doctest::

   >>> Vh = fem.Basis(mesh, fem.ElementTriP1())

第五步: 装配线性方程组
==================================

现在有限元装配万事俱备.
产生的矩阵具有 ``scipy.sparse.csr_matrix`` 类型.

.. doctest::

   >>> A = a.assemble(Vh)
   >>> l = L.assemble(Vh)
   >>> A.shape
   (81, 81)
   >>> l.shape
   (81,)

第六步: 找边界DOFs.
==========================

设置边界条件需要找到矩阵 :math:`A` 的行和列， 使它与边上的自由度(DOFs)相匹配。

默认情况下, :meth:`~skfem.assembly.CellBasis.get_dofs` 匹配所有的边界面并找出相应DOFs.

.. doctest::

   >>> D = Vh.get_dofs()
   >>> D.flatten()
   array([ 0,  1,  2,  3,  4,  5,  7,  8,  9, 10, 11, 13, 14, 16, 17, 18, 25,
          26, 27, 29, 30, 32, 33, 34, 35, 36, 39, 40, 49, 50, 53, 54])

第七步: 删除边界DOFs.
===============================

为了边界上设置:math:`u=0`，必须从线性方程组 :math:`Ax=l` 中删除边界自由度.
This can be done using :func:`~skfem.utils.condense`.

.. doctest::

   >>> system = fem.condense(A, l, D=D)
   >>> system[0].shape
   (49, 49)
   >>> system[1].shape
   (49,)

第八步: 求解线性方程组
===============================

通过请求 :func:`~skfem.utils.solve`可以求解稠密线性方程组。
 :func:`~skfem.utils.solve` 是``scipy.sparse.linalg``中的简单包装器。
 结果自动与原方程组的维数相匹配。

.. doctest::

   >>> x = fem.solve(*system)
   >>> x.shape
   (81,)


第九步: 计算误差
=======================

本例中精确解为

.. math::

   u(x, y) = \frac{1}{2 \pi^2} \sin \pi x \sin \pi y.

因此, 误差很小讲得通.

.. doctest::

   >>> @fem.Functional
   ... def error(w):
   ...     x, y = w.x
   ...     uh = w['uh']
   ...     u = np.sin(np.pi * x) * np.sin(np.pi * y) / (2. * np.pi ** 2)
   ...     return (uh - u) ** 2
   >>> error.assemble(Vh, uh=Vh.interpolate(x))
   1.069066819861505e-06
