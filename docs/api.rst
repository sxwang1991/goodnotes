==========================
 应用程序接口API细节描述
==========================

本节包含软件包最常用的界面API文档。

模块: skfem.mesh
==================

.. automodule:: skfem.mesh

抽象类: Mesh
--------------------

.. autoclass:: skfem.mesh.Mesh
   :members: load, save, refined, element_finder, doflocs, t

类: MeshTri
**************

.. autoclass:: skfem.mesh.MeshTri

.. autoclass:: skfem.mesh.MeshTri1
   :members: __init__, init_symmetric, init_sqsymmetric, init_refdom, init_tensor, init_lshaped, init_circle, load

.. autoclass:: skfem.mesh.MeshTri2
   :members: init_circle, load

类: MeshQuad
***************

.. autoclass:: skfem.mesh.MeshQuad

.. autoclass:: skfem.mesh.MeshQuad1
   :members: __init__, init_refdom, init_tensor, to_meshtri, load

.. autoclass:: skfem.mesh.MeshQuad2
   :members: load

类: MeshTet
**************

.. autoclass:: skfem.mesh.MeshTet

.. autoclass:: skfem.mesh.MeshTet1
   :members: __init__, init_refdom, init_tensor, init_ball, load

.. autoclass:: skfem.mesh.MeshTet2
   :members: init_ball, load

类: MeshHex
**************

.. autoclass:: skfem.mesh.MeshHex

.. autoclass:: skfem.mesh.MeshHex1
   :members: __init__, init_tensor, to_meshtet, load

类: MeshLine
***************

.. autoclass:: skfem.mesh.MeshLine

.. autoclass:: skfem.mesh.MeshLine1
   :members: __init__

模块: skfem.assembly
======================

.. automodule:: skfem.assembly

抽象类: AbstractBasis
-----------------------------

 :class:`~skfem.assembly.basis.AbstractBasis` 的子类代表总体有限元基函数在积分点出的值。

类: CellBasis
****************

.. autoclass:: skfem.assembly.Basis

.. autoclass:: skfem.assembly.CellBasis
   :members: __init__


类: BoundaryFacetBasis
*************************

.. autoclass:: skfem.assembly.FacetBasis

.. autoclass:: skfem.assembly.BoundaryFacetBasis
   :members: __init__, trace

类: InteriorFacetBasis
*************************

.. autoclass:: skfem.assembly.InteriorFacetBasis
   :members: __init__


抽象类: Form
--------------------

类: BilinearForm
*******************

.. autoclass:: skfem.assembly.BilinearForm
   :members: assemble

类: LinearForm
*****************

.. autoclass:: skfem.assembly.LinearForm
   :members: assemble

类: Functional
*****************

.. autoclass:: skfem.assembly.Functional
   :members: assemble, elemental

模块: skfem.element
=====================

.. automodule:: skfem.element
   :members:
   :show-inheritance:
   :exclude-members: DiscreteField, ElementVectorH1

模块: skfem.utils
===================

函数: solve
---------------

.. autofunction:: skfem.utils.solve

函数: condense
------------------

.. autofunction:: skfem.utils.condense

函数: enforce
-----------------

.. autofunction:: skfem.utils.enforce

函数: projection
--------------------

.. autofunction:: skfem.utils.projection

模块: skfem.helpers
=====================

.. automodule:: skfem.helpers

.. autofunction:: skfem.helpers.grad

.. autofunction:: skfem.helpers.div

.. autofunction:: skfem.helpers.curl

.. autofunction:: skfem.helpers.d

.. autofunction:: skfem.helpers.dd

.. autofunction:: skfem.helpers.sym_grad

.. autofunction:: skfem.helpers.dot

.. autofunction:: skfem.helpers.ddot
