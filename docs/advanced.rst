===============
进阶议题
===============

为了更加细致的理解这个软件包这目标，本节围绕scikit-fem的特征作进阶讨论。

.. _forms:

形式的细细剖析
================

我们考虑形式作为有限元装配的最基本的组建模块。
因此，理解scikit-fem中形式是如何使用、如何准确表达是很重要的。

我们以例子开始。  

Laplace 算子 :math:`-\Delta` 相应的双线性形式为

.. math::

   a(u, v) = \int_\Omega \nabla u \cdot \nabla v \,\mathrm{d}x.

为了在scikit-fem中表达这一形式, 将被积函数写成Python函数:

.. doctest::

   >>> from skfem import BilinearForm
   >>> from skfem.helpers import grad, dot
   >>> @BilinearForm
   ... def integrand(u, v, w):
   ...    return dot(grad(u), grad(v))

给定函数的 :math:`L^2` 内积和测试函数 :math:`v` 所给出,典型的载荷向量, e.g.,

.. math::

   b(v) = \int_\Omega \sin(\pi x) \sin(\pi y) v \,\mathrm{d}x.

这可以写成

.. doctest::

   >>> import numpy as np
   >>> from skfem import LinearForm
   >>> @LinearForm
   ... def loading(v, w):
   ...    return np.sin(np.pi * w.x[0]) * np.sin(np.pi * w.x[1]) * v

另外, 形式依赖于局部网格参数 ``w.h`` 或者其他有限元函数(见 :ref:`predefined`) 。
而且, 边界形式可以依赖于法向量 ``w.n`` 。
一个例子是形式

.. math::

   l(\boldsymbol{v}) = \int_{\partial \Omega} \boldsymbol{v} \cdot \boldsymbol{n} \,\mathrm{d}s

它可以写成

.. doctest::

   >>> from skfem import LinearForm
   >>> from skfem.helpers import dot
   >>> @LinearForm
   ... def loading(v, w):
   ...    return dot(w.n, v)

辅助函数(helpers)比如 ``dot`` 在 :ref:`helpers` 中作进一步讨论.

.. _formsreturn:

形式返回NumPy数组
-------------------------

形式定义总要返回一个二维NumPy数组。

这可以使用Python调试程序得以验证:

.. code-block:: python

   from skfem import *
   from skfem.helpers import grad, dot
   @BilinearForm
   def integrand(u, v, w):
       import pdb; pdb.set_trace()  # breakpoint
       return dot(grad(u), grad(v))

将上述程序片段保存为 ``test.py`` 并通过 ``python -i test.py`` 运行它。
允许作试验:

.. code-block:: none

   tom@tunkki:~/src/scikit-fem$ python -i test.py
   >>> asm(integrand, Basis(MeshTri(), ElementTriP1()))
   > /home/tom/src/scikit-fem/test.py(7)integrand()
   -> return dot(grad(u), grad(v))
   (Pdb) dot(grad(u), grad(v))
   array([[2., 2., 2.],
          [1., 1., 1.]])

注意 ``dot(grad(u), grad(v))`` 是维数为 ` 单元数 ` x ` 每个单元上积分点数 `的NumPy数组。  

不管使用哪一个种网格或单元类型，这个返回值的维数总是这样。

.. _helpers:

辅助函数(helpers)很有用但并非必要
------------------------------------

模块 :mod:`skfem.helpers` 包含使形式更加具有可读性的函数。

上述形式的另一种写法是

.. doctest:: python

   >>> from skfem import BilinearForm
   >>> @BilinearForm
   ... def integrand(u, v, w):
   ...     return u[1][0] * v[1][0] + u[1][1] * v[1][1]

.. note::

    事实上, ``u`` 和 ``v`` 是NumPy数组的简单元组，``u[0]`` 处取函数值，``u[1]`` 处取梯度值(还有别的魔力比如执行 ``__array__`` 与 ``__mul__`` 如此这般可以使得 ``u * v`` 能按预期那样运行)。

注意 ``u[0]`` 的维数也如从 :ref:`formsreturn`: 讨论中我们预期的那样的返回值。 

.. code-block:: none

   tom@tunkki:~/src/scikit-fem$ python -i test.py
   >>> asm(integrand, Basis(MeshTri(), ElementTriP1()))
   > /home/tom/src/scikit-fem/test.py(7)integrand()
   -> return dot(grad(u), grad(v))
   (Pdb) !u[0]
   array([[0.66666667, 0.16666667, 0.16666667],
          [0.66666667, 0.16666667, 0.16666667]])


.. _dofindexing:

自由度的指标
==================================

.. warning::

   本节包含自由度的排序更加低阶别之细节。
   只有在你在 :ref:`finddofs` 中没找到答案之时方可阅读本节。

自由度 :math:`x` 基于网格和单元类型自动排序。  
手动调查自由度是如何匹配不同网格的拓扑实体 (`nodes`, `facets`, `edges`, `elements`) 也是可行的。


.. note::

   **术语:** 在scikit-fem中, `edges` 仅仅存在与三维网格中，以至于 `facets` 总是共享单元的公共部分.  特别地, 我们把三角形网格和四边形网格的边称作 `facets`.

例如, 考虑二次Lagrange三角形元和默认单位正方形二次元网格：

.. doctest::

   >>> from skfem import *
   >>> m = MeshTri()
   >>> m
   Triangular mesh with 4 vertices and 2 elements.
   >>> basis = Basis(m, ElementTriP2())

相应于网格节点 (或顶点)的自由度是

.. doctest::

   >>> basis.nodal_dofs
   array([[0, 1, 2, 3]])

上面第一列相应于在相应网格数据结构中的第一列:

.. doctest::

   >>> m.p
   array([[0., 1., 0., 1.],
          [0., 0., 1., 1.]])

特别地, 在 :math:`(0,0)` 处节点相应于向量 :math:`x` 的第1个元素, 
在 :math:`(1,0)` 处节点相应于向量 :math:`x` 的第2个元素, 以此类推。

类似地, 相应于网格边(facet)的自由度为

.. doctest::

   >>> basis.facet_dofs
   array([[4, 5, 6, 7, 8]])

相应的边(facet)可以在网格数据结构中找到:

.. doctest::

   >>> m.facets
   array([[0, 0, 1, 1, 2],
          [1, 2, 2, 3, 3]])
   >>> .5 * m.p[:, m.facets].sum(axis=0)  # 边(facet)的中点
   array([[0. , 0. , 0.5, 0.5, 0.5],
          [0.5, 0.5, 0.5, 1. , 1. ]])
   
每一个自由度 
要么节点 (``nodal_dofs``), 界面 (``facet_dofs``), 边 (``edge_dofs``), 要么与单元 (``interior_dofs``) 有关。
