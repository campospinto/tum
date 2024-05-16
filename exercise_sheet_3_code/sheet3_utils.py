import numpy                as np
import matplotlib.pyplot    as plt
from   matplotlib           import cm, colors

from scipy.sparse   import dia_matrix

from sympde.expr                import BilinearForm, integral, EssentialBC
from sympde.topology            import Line, elements_of, Derham, element_of
from sympde.topology.datatype   import H1Space, HcurlSpace, L2Space

from psydac.api.discretization      import discretize
from psydac.api.essential_bc        import apply_essential_bc
from psydac.linalg.basic            import LinearOperator
from psydac.linalg.block            import BlockVectorSpace, BlockLinearOperator
from psydac.linalg.direct_solvers   import BandedSolver
from psydac.linalg.kron             import KroneckerLinearSolver
from psydac.linalg.stencil          import StencilMatrix

def get_diagonal(A, inv=True):
    """
    Takes a LinearOperator (similar to a mass matrix of a 2D VectorFunctionSpace) and returns a BlockLinearOperator object.
    This BlockLinearOperator is a diagonal matrix, and its diagonal entries are those, or those inverted, of the LinearOperator A.
    
    """
    assert isinstance(inv, bool)
    assert isinstance(A, LinearOperator)

    V   = A.domain
    v   = V.zeros()

    Ac = BlockLinearOperator(V, V)

    for b_index in range(V.n_blocks):
        diag_values = []

        V_block = V[b_index]

        npts1, npts2 = V_block.npts
        pads1, pads2 = V_block.pads

        Ac[b_index, b_index] = StencilMatrix(V_block, V_block)
        Ac_block   = Ac[b_index, b_index]

        for n1 in range(npts1):
            diag_values_block = []
            for n2 in range(npts2):
                v *= 0.0
                v[b_index]._data[pads1+n1, pads2+n2] = 1
                w = A @ v
                out = w[b_index]._data[pads1+n1, pads2+n2]
                if inv:
                    out = 1/out
                diag_values_block.append(out)
            diag_values.append(diag_values_block)
        diag_values = np.array(diag_values)
        diagonal_indices = Ac_block._get_diagonal_indices()
        Ac_block._data[diagonal_indices] = diag_values
    
    return Ac

def plot(gridsize_x, title, funs, titles, gridsize_y=None, surface_plot=False):

    if gridsize_y is None:
        x = np.linspace(0, 1, gridsize_x+1)
        vals = [[] for fun in funs]
        for i, fun in enumerate(funs):
            for xi in x:
                vals[i].append(fun(xi))
        
        n_plots = len(funs)
        if n_plots > 1:
            assert n_plots == len(titles)
        else:
            print('Warning [plot]: will discard argument titles for a single plot')

        fig = plt.figure(figsize=(2.6+4.8*n_plots, 4.8))
        fig.suptitle(title, fontsize=14)

        for i in range(n_plots):
            ax = fig.add_subplot(1, n_plots, i+1)
            ax.plot(x, vals[i])
            ax.set_xlabel(r'$x$')
            if n_plots > 1:
                ax.set_title ( titles[i] )
        plt.show()
    elif gridsize_y is not None:
        x = np.linspace(0, 1, gridsize_x+1)
        y = np.linspace(0, 1, gridsize_y+1)
        xx, yy = np.meshgrid(x, y)
        vals = [[] for fun in funs]
        for i, fun in enumerate(funs):
            for xi, yi in zip(xx, yy):
                vals[i].append([fun(xii, yii) for xii, yii in zip(xi, yi)])
            vals[i] = np.array(vals[i])

        n_plots = len(funs)
        if n_plots > 1:
            assert n_plots == len(titles)
        else:
            if titles:
                print('Warning [plot]: will discard argument titles for a single plot')

        fig = plt.figure(figsize=(2.6+4.8*n_plots, 4.8))
        fig.suptitle(title, fontsize=14)

        for i in range(n_plots):
            vmin = np.min(vals[i])
            vmax = np.max(vals[i])
            cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
            ax = fig.add_subplot(1, n_plots, i+1)
            ax.contourf(xx, yy, vals[i], 50, norm=cnorm, cmap='viridis')
            ax.axis('equal')
            fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap='viridis'), ax=ax,  pad=0.05)
            ax.set_xlabel( r'$x$', rotation='horizontal' )
            ax.set_ylabel( r'$y$', rotation='horizontal' )
            if n_plots > 1:
                ax.set_title ( titles[i] )
        plt.show()

        if surface_plot:
            fig = plt.figure(figsize=(2.6+4.8*n_plots, 4.8))
            fig.suptitle(title+' -- surface', fontsize=14)

            for i in range(n_plots):
                vmin = np.min(vals[i])
                vmax = np.max(vals[i])
                cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
                ax = fig.add_subplot(1, n_plots, i+1, projection='3d')
                ax.plot_surface(xx, yy, vals[i], norm=cnorm, cmap='viridis',
                            linewidth=0, antialiased=False)
                fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap='viridis'), ax=ax,  pad=0.05)
                ax.set_xlabel( r'$x$', rotation='horizontal' )
                ax.set_ylabel( r'$y$', rotation='horizontal' )
                if n_plots > 1:
                    ax.set_title ( titles[i] )
            plt.show()

class DirichletLinearOperator_2D(LinearOperator):
    """ A linear operator additionally applying homogeneous (essential) Dirichlet BCs before its evaluation. 
    Upon transposing, the BCs will be enforced after applying the transpose of the linear operator. This can be done by passing the flag `transpose`.
    Upon choosing mass_matrix=True, changes a mass matrix M to behave as P_0 @ M @ P_0 + P_Gamma @ M @ P_Gamma,
    which corresponds to making M block-diagonal in a suitably chosen basis. P_0 projects into the interior degrees of freedom, P_Gamma = I - P_0 projects
    into the exterior degrees of freedom.

    Parameters
    ----------
    A : LinearOperator
    domain_space : continuous function space corresponding to the domain of A; must have kind property
    codomain_space : continuous function space corresponding to the codomain of A; must have kind property
    periodic : list of 2 booleans, periodicity in x and y directions; optional

    """
    def __init__(self, A, domain_space, codomain_space, periodic=[False, False], mass_matrix=False, transpose=False):
        
        assert isinstance(mass_matrix, bool)
        self._A = A
        self._domain = A.domain
        self._codomain = A.codomain
        self._domain_space = domain_space
        self._codomain_space = codomain_space
        self._periodic = periodic
        self._mass_matrix = mass_matrix
        self._transpose = transpose
        if mass_matrix:
            assert self._domain == self._codomain
            assert self._domain_space == self._codomain_space
            self._v_copy_2 = self._domain.zeros()
            self._w_2 = self._domain.zeros()
            self._w_2_copy = self._domain.zeros()
        self._v_copy = self._domain.zeros()
        self._BC = self._get_BCs()

    def _get_BCs(self):
        """
        Consider the 2D de Rham sequence

                grad           curl
        V_h^0 -------> V_h^1 -------> V_h^2
          |             |               |
          | \sigma_0    | \sigma_1      | \sigma_2
          |             |               |
          v      G      v       C       v
        \cC^0 -------> \cC^1 -------> \cC^2

        with

        V_h^0 \subset H^1,
        V_h^1 \subset H(curl),
        V_h^2 \subset L^2.

        Provided a situation in which we want to solve for a function satisfying essential Dirichlet BCs,
        we replace the differential matrices G and C with

        G_0     = G @ P0_0,
        C_0     = C @ P1_0,

        and respectively

        G_0.T   = P0_0 @ G.T,
        C_0.T   = P1_0 @ C.T.

        Here, P0_0 and P1_0 are projections into the interior coefficients of V_h^0 and V_h^1 respectively.

        Consequently, if self._transpose == True, we must determine the BCs of the codomain space of self._A.
        If self._transpose == False, we must determine the BCs of the domain space of self._A.
        
        """
        
        transpose = self._transpose
        if transpose == True:
            space = self._codomain_space
        else:
            space = self._domain_space
        
        periodic = self._periodic

        assert all([isinstance(P, bool) for P in periodic])
        if all([P == True for P in periodic]):
            return None
        
        u = element_of(space, name='u')
        bcs = [EssentialBC(u, 0, side, position=0) for side in space.domain.boundary]

        if space.kind == H1Space:
            bcs_x = [bcs[0], bcs[1]]
            bcs_y = [bcs[2], bcs[3]]
        elif space.kind == HcurlSpace:
            bcs_x = [bcs[2], bcs[3]]
            bcs_y = [bcs[0], bcs[1]]
        
        for P,bc in zip(periodic, (bcs_x, bcs_y)):
            if P:
                bc *= 0

        if space.kind == H1Space:
            BC = bcs_x + bcs_y
        elif space.kind == HcurlSpace:
            BC = [bcs_x, bcs_y]
        elif space.kind == L2Space:
            BC = None
        else:
            raise ValueError('Domain Space kind not recognized')

        return BC

    @property
    def domain(self):
        return self._domain
    
    @property
    def codomain(self):
        return self._codomain
    
    @property
    def dtype(self):
        return self._A.dtype
    
    def tosparse(self):
        raise NotImplementedError
    
    def toarray(self):
        raise NotImplementedError
    
    def transpose(self, conjugate=False):
        return DirichletLinearOperator_2D(self._A.T, self._codomain_space, self._domain_space, self._periodic, mass_matrix=self._mass_matrix, transpose=not self._transpose)
    
    def dot(self, v, out=None):
        """
        Projects v into the subspace of hom. DBCs of the domain function space and then applies A to the projected coefficient vector.
        Result is written to Vector out, if provided.

        If self._transpose == True, instead applies hom. DBCs of the codomain function space to the output of self._A @ v.

        If self._mass_matrix == True, writing M = self._A and denoting with P_0 and P_Gamma projections into the 
        interior and exterior coefficient space respectively, the dot product as if v were applied to
        P_0 @ M @ P_0 + P_Gamma @ M @ P_Gamma.
        
        """

        # not a mass matrix (a differential matrix)
        if not self._mass_matrix:
            BC = self._BC
            
            # BCs not None (not M2 mass matrix)
            if BC is not None:
                # not transposed
                if not self._transpose:
                    vc = self._v_copy
                    v.copy(out=vc)
                    # BlockVector
                    if isinstance(self._A.domain, BlockVectorSpace): 
                        for vi, bci in zip(vc, BC):
                            apply_essential_bc(vi, *bci)
                    # StencilVector
                    else:
                        apply_essential_bc(vc, *BC)
                    w = self._A.dot(vc, out=out)
                # transposed
                else:
                    w = self._A.dot(v, out=out)
                    # BlockVector
                    if isinstance(self._A.codomain, BlockVectorSpace): 
                        for wi, bci in zip(w, BC):
                            apply_essential_bc(wi, *bci)
                    # StencilVector
                    else:
                        apply_essential_bc(w, *BC)
            # BCs are None (M2 mass matrix)
            else:
                w = self._A.dot(v, out=out)
        # a mass matrix
        else:
            BC = self._BC

            # BCs not None (not M2 mass matrix)
            if BC is not None:
                """
                Compute [ P_0 @ M @ P_0 + P_Gamma @ M @ P_Gamma ] @ v

                1. Addend:
                    Copy v into vc; Apply BCs to vc; Apply M to vc, store in w; Apply BCs to w
                2. Addend:
                    Copy v into vc2; Substract vc from vc2 after applying BCs to obtain P_Gamma @ v;
                    Apply M to vc2, store in w2; Copy w2 into w2c; Apply BCs to w2c;
                    Substract w2c from w2 to obtain P_Gamma @ M @ P_Gamma @ v
                3. Add
                    Add w2 to w to obtain [ P_0 @ M @ P_0 + P_Gamma @ M @ P_Gamma ] @ v
                
                """
                vc  = self._v_copy
                vc2 = self._v_copy_2
                w2  = self._w_2
                w2c = self._w_2_copy

                v.copy(out=vc)
                v.copy(out=vc2)

                # BlockVector
                if isinstance(self._A.domain, BlockVectorSpace): 
                    for vi, bci in zip(vc, BC):
                        apply_essential_bc(vi, *bci)
                # StencilVector
                else:
                    apply_essential_bc(vc, *BC)

                vc2 -= vc

                w = self._A.dot(vc, out=out)
                self._A.dot(vc2, out=w2)

                w2.copy(out=w2c)

                # BlockVector
                if isinstance(self._A.domain, BlockVectorSpace): 
                    for wi, bci in zip(w, BC):
                        apply_essential_bc(wi, *bci)
                    for w2i, bci in zip(w2c, BC):
                        apply_essential_bc(w2i, *bci)
                # StencilVector
                else:
                    apply_essential_bc(w, *BC)
                    apply_essential_bc(w2c, *BC)
                w2 -= w2c
                w += w2
            # BCs are None (M2 mass matrix)
            else:
                w = self._A.dot(v, out=out)

        return w

def to_bnd(A):

    dmat = dia_matrix(A.toarray(), dtype=A.dtype)
    la   = abs(dmat.offsets.min())
    ua   = dmat.offsets.max()
    cmat = dmat.tocsr()

    A_bnd = np.zeros((1+ua+2*la, cmat.shape[1]), A.dtype)

    for i,j in zip(*cmat.nonzero()):
        A_bnd[la+ua+i-j, j] = cmat[i,j]

    return A_bnd, la, ua

def matrix_to_bandsolver(A):
    A.remove_spurious_entries()
    A_bnd, la, ua = to_bnd(A)
    return BandedSolver(ua, la, A_bnd)

def get_M1_block_kron_solver_2D(V1, ncells, degree, periodic):
    """
    Given a 2D DeRham sequenece (V0 = H(grad) --grad--> V1 = H(curl) --curl--> V2 = L2)
    discreticed using ncells, degree and periodic,

        domain = Square('C', bounds1=(0, 1), bounds2=(0, 1))
        derham = Derham(domain)
        domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
        derham_h = discretize(derham, domain_h, degree=degree),

    returns the inverse of the mass matrix M1 as a BlockLinearOperator consisting of two KroneckerLinearSolvers on the diagonal.
    """
    # assert 3D
    assert len(ncells) == 2
    assert len(degree) == 2
    assert len(periodic) == 2

    # 1D domain to be discreticed using the respective values of ncells, degree, periodic
    domain_1d = Line('L', bounds=(0,1))
    derham_1d = Derham(domain_1d)

    # storage for the 1D mass matrices
    M0_matrices = []
    M1_matrices = []

    # assembly of the 1D mass matrices
    for (n, p, P) in zip(ncells, degree, periodic):

        domain_1d_h = discretize(domain_1d, ncells=[n], periodic=[P])
        derham_1d_h = discretize(derham_1d, domain_1d_h, degree=[p])

        u_1d_0, v_1d_0 = elements_of(derham_1d.V0, names='u_1d_0, v_1d_0')
        u_1d_1, v_1d_1 = elements_of(derham_1d.V1, names='u_1d_1, v_1d_1')

        a_1d_0 = BilinearForm((u_1d_0, v_1d_0), integral(domain_1d, u_1d_0 * v_1d_0))
        a_1d_1 = BilinearForm((u_1d_1, v_1d_1), integral(domain_1d, u_1d_1 * v_1d_1))

        a_1d_0_h = discretize(a_1d_0, domain_1d_h, (derham_1d_h.V0, derham_1d_h.V0))
        a_1d_1_h = discretize(a_1d_1, domain_1d_h, (derham_1d_h.V1, derham_1d_h.V1))

        M_1d_0 = a_1d_0_h.assemble()
        M_1d_1 = a_1d_1_h.assemble()

        M0_matrices.append(M_1d_0)
        M1_matrices.append(M_1d_1)

    V1_1 = V1[0]
    V1_2 = V1[1]

    B1_mat = [M1_matrices[0], M0_matrices[1]]
    B2_mat = [M0_matrices[0], M1_matrices[1]]

    B1_solvers = [matrix_to_bandsolver(Ai) for Ai in B1_mat]
    B2_solvers = [matrix_to_bandsolver(Ai) for Ai in B2_mat]

    B1_kron_inv = KroneckerLinearSolver(V1_1, V1_1, B1_solvers)
    B2_kron_inv = KroneckerLinearSolver(V1_2, V1_2, B2_solvers)

    M1_block_kron_solver = BlockLinearOperator(V1, V1, ((B1_kron_inv, None), 
                                                        (None, B2_kron_inv)))

    return M1_block_kron_solver

def get_M1_block_kron_solver(V1, ncells, degree, periodic):
    """
    Given a 3D DeRham sequenece (V0 = H(grad) --grad--> V1 = H(curl) --curl--> V2 = H(div) --div--> V3 = L2)
    discreticed using ncells, degree and periodic,

        domain = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
        derham = Derham(domain)
        domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
        derham_h = discretize(derham, domain_h, degree=degree),

    returns the inverse of the mass matrix M1 as a BlockLinearOperator consisting of three KroneckerLinearSolvers on the diagonal.
    """
    # assert 3D
    assert len(ncells) == 3
    assert len(degree) == 3
    assert len(periodic) == 3

    # 1D domain to be discreticed using the respective values of ncells, degree, periodic
    domain_1d = Line('L', bounds=(0,1))
    derham_1d = Derham(domain_1d)

    # storage for the 1D mass matrices
    M0_matrices = []
    M1_matrices = []

    # assembly of the 1D mass matrices
    for (n, p, P) in zip(ncells, degree, periodic):

        domain_1d_h = discretize(domain_1d, ncells=[n], periodic=[P])
        derham_1d_h = discretize(derham_1d, domain_1d_h, degree=[p])

        u_1d_0, v_1d_0 = elements_of(derham_1d.V0, names='u_1d_0, v_1d_0')
        u_1d_1, v_1d_1 = elements_of(derham_1d.V1, names='u_1d_1, v_1d_1')

        a_1d_0 = BilinearForm((u_1d_0, v_1d_0), integral(domain_1d, u_1d_0 * v_1d_0))
        a_1d_1 = BilinearForm((u_1d_1, v_1d_1), integral(domain_1d, u_1d_1 * v_1d_1))

        a_1d_0_h = discretize(a_1d_0, domain_1d_h, (derham_1d_h.V0, derham_1d_h.V0))
        a_1d_1_h = discretize(a_1d_1, domain_1d_h, (derham_1d_h.V1, derham_1d_h.V1))

        M_1d_0 = a_1d_0_h.assemble()
        M_1d_1 = a_1d_1_h.assemble()

        M0_matrices.append(M_1d_0)
        M1_matrices.append(M_1d_1)

    V1_1 = V1[0]
    V1_2 = V1[1]
    V1_3 = V1[2]

    B1_mat = [M1_matrices[0], M0_matrices[1], M0_matrices[2]]
    B2_mat = [M0_matrices[0], M1_matrices[1], M0_matrices[2]]
    B3_mat = [M0_matrices[0], M0_matrices[1], M1_matrices[2]]

    B1_solvers = [matrix_to_bandsolver(Ai) for Ai in B1_mat]
    B2_solvers = [matrix_to_bandsolver(Ai) for Ai in B2_mat]
    B3_solvers = [matrix_to_bandsolver(Ai) for Ai in B3_mat]

    B1_kron_inv = KroneckerLinearSolver(V1_1, V1_1, B1_solvers)
    B2_kron_inv = KroneckerLinearSolver(V1_2, V1_2, B2_solvers)
    B3_kron_inv = KroneckerLinearSolver(V1_3, V1_3, B3_solvers)

    M1_block_kron_solver = BlockLinearOperator(V1, V1, ((B1_kron_inv, None, None), 
                                                              (None, B2_kron_inv, None), 
                                                              (None, None, B3_kron_inv)))

    return M1_block_kron_solver

def get_M2_block_kron_solver(V2, ncells, degree, periodic):
    """
    Given a 3D DeRham sequenece (V0 = H(grad) --grad--> V1 = H(curl) --curl--> V2 = H(div) --div--> V3 = L2)
    discreticed using ncells, degree and periodic,

        domain      = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
        derham      = Derham(domain)
        domain_h    = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
        derham_h    = discretize(derham, domain_h, degree=degree),

    returns the inverse of the mass matrix M2 as a BlockLinearOperator consisting of three KroneckerLinearSolvers on the diagonal.
    """
    # assert 3D
    assert len(ncells) == 3
    assert len(degree) == 3
    assert len(periodic) == 3

    # 1D domain to be discreticed using the respective values of ncells, degree, periodic
    domain_1d = Line('L', bounds=(0,1))
    derham_1d = Derham(domain_1d)

    # storage for the 1D mass matrices
    M0_matrices = []
    M1_matrices = []

    # assembly of the 1D mass matrices
    for (n, p, P) in zip(ncells, degree, periodic):

        domain_1d_h = discretize(domain_1d, ncells=[n], periodic=[P])
        derham_1d_h = discretize(derham_1d, domain_1d_h, degree=[p])

        u_1d_0, v_1d_0 = elements_of(derham_1d.V0, names='u_1d_0, v_1d_0')
        u_1d_1, v_1d_1 = elements_of(derham_1d.V1, names='u_1d_1, v_1d_1')

        a_1d_0 = BilinearForm((u_1d_0, v_1d_0), integral(domain_1d, u_1d_0 * v_1d_0))
        a_1d_1 = BilinearForm((u_1d_1, v_1d_1), integral(domain_1d, u_1d_1 * v_1d_1))

        a_1d_0_h = discretize(a_1d_0, domain_1d_h, (derham_1d_h.V0, derham_1d_h.V0))
        a_1d_1_h = discretize(a_1d_1, domain_1d_h, (derham_1d_h.V1, derham_1d_h.V1))

        M_1d_0 = a_1d_0_h.assemble()
        M_1d_1 = a_1d_1_h.assemble()

        M0_matrices.append(M_1d_0)
        M1_matrices.append(M_1d_1)

    V2_1 = V2[0]
    V2_2 = V2[1]
    V2_3 = V2[2]

    B1_mat = [M0_matrices[0], M1_matrices[1], M1_matrices[2]]
    B2_mat = [M1_matrices[0], M0_matrices[1], M1_matrices[2]]
    B3_mat = [M1_matrices[0], M1_matrices[1], M0_matrices[2]]

    B1_solvers = [matrix_to_bandsolver(Ai) for Ai in B1_mat]
    B2_solvers = [matrix_to_bandsolver(Ai) for Ai in B2_mat]
    B3_solvers = [matrix_to_bandsolver(Ai) for Ai in B3_mat]

    B1_kron_inv = KroneckerLinearSolver(V2_1, V2_1, B1_solvers)
    B2_kron_inv = KroneckerLinearSolver(V2_2, V2_2, B2_solvers)
    B3_kron_inv = KroneckerLinearSolver(V2_3, V2_3, B3_solvers)

    M2_block_kron_solver = BlockLinearOperator(V2, V2, ((B1_kron_inv, None, None), 
                                                              (None, B2_kron_inv, None), 
                                                              (None, None, B3_kron_inv)))

    return M2_block_kron_solver