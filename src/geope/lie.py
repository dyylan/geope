import numpy as np
import scipy.linalg as spla
import itertools as it
import re
import jax
import jax.numpy as jnp

from . import utils


class Basis:
    def __init__(self, 
                 basis: np.ndarray, 
                 labels: list = [], 
                 local_dim: int = 2, 
                 interaction_graph: list = None,
                 interaction_map: dict = None):
        assert basis.ndim == 3, '`basis` must be a rank 3 tensor'
        # assert (basis.shape[1] == basis.shape[2]) and (np.emath.logn(local_dim, basis.shape[1]) == int(np.emath.logn(local_dim, basis.shape[1]))), \
        #     '`basis` must be a tensor of shape (n, 2**n, 2**n), where n corresponds to the matrix dimension, ' \
        #     f'received {basis.shape}'
        self._basis = basis
        self._labels = labels
        self._plot_labels = self._generate_plot_labels()
        self._interaction_labels = self._generate_interaction_labels()
        self._interaction_qubits = self._generate_interaction_qubits()
        self._interaction_graph = self.apply_interaction_graph(interaction_graph) if interaction_graph is not None else self._generate_interaction_graph()
        self._interaction_map = self.apply_interaction_map(interaction_map) if interaction_map is not None else self._generate_interaction_map()
        self._local_dim = local_dim
        self._dim = basis.shape[1]
        self._lie_algebra_dim = basis.shape[0]
        self._n = int(np.log2(basis.shape[1]))
        assert self._n

    def linear_span(self, parameters):
        parameters = np.reshape(parameters, (-1, 1, 1))
        return np.einsum('nij,nij->ij', parameters, self._basis)

    def overlap(self, other):
        out = utils.traces(self.basis, other.basis)
        return ~np.isclose(np.sum(out, axis=0), 0)

    def verify(self):
        out = utils.traces(self.basis, self.basis)
        return np.allclose(np.diag(np.diag(out)), out)

    def apply_interaction_graph(self, interaction_graph: list):
        """
        Applies an interaction graph to the basis. The interaction graph is a list of tuples, where each tuple
        contains the indices of the qubits that interact with each other.

        Parameters
        ----------
        interaction_graph : list
            A list of tuples or lists representing the interaction graph.
        """
        interaction_graph = [tuple(interaction) for interaction in interaction_graph]
        self._interaction_graph = interaction_graph
        del_indices = []
        for i, interaction in enumerate(self.interaction_qubits):
            if (interaction not in interaction_graph) and (len(interaction)>1):
                del_indices.append(i)
        self._remove_basis_elements(del_indices)
        return interaction_graph
    
    def apply_interaction_map(self, interaction_map: dict):
        """
        Applies an interaction map to the basis. The interaction map is a dictionary where the keys are tuples
        representing the qubits interacting, and the values are lists of interactions corresponding
        to the interactions between those qubits.

        Parameters
        ----------
        interaction_map : dict
            A dictionary representing the interaction map.
        """
        self._interaction_map = interaction_map
        del_indices = []
        for i, interaction in enumerate(self.interaction_qubits):
            if (interaction not in interaction_map.keys()):
                del_indices.append(i)
            elif self.interaction_labels[i] not in interaction_map[interaction]:
                del_indices.append(i)
        self._remove_basis_elements(del_indices)
        return interaction_map

    def _generate_plot_labels(self):
        if self.labels:
            new_labels = []
            for label in self.labels:
                new_label = "$"
                for i,c in enumerate(label):
                    new_label += "" if c=="I" else f"{c}_{{{i+1}}}"
                new_label += "$"
                new_labels.append(new_label)
            return new_labels
        else:
            return None

    def _generate_interaction_labels(self):
        if self.labels:
            new_labels = []
            for label in self.labels:
                new_label = ""
                for c in label:
                    new_label += "" if c=="I" else f"{c}".lower()
                new_labels.append(new_label)
            return new_labels
        else:
            return None

    def _generate_interaction_qubits(self):
        if self.labels:
            interaction_qubits = []
            for label in self.plot_labels:
                qubits = re.findall(r'\d+', label)
                interaction_qubits.append(tuple([int(q) for q in qubits]))
            return interaction_qubits
        else:
            return None

    def _generate_interaction_graph(self):
        interaction_graph = []
        for interaction in self.interaction_qubits:
            if len(interaction) > 1:
                interaction_graph.append(interaction)
        return interaction_graph

    def _generate_interaction_map(self):
        interaction_map = {}
        for i, interaction in enumerate(self.interaction_qubits):
            interaction_map.setdefault(interaction, []).append(self.interaction_labels[i])
        return interaction_map

    def _remove_basis_elements(self, indices):
        for i in sorted(indices, reverse=True):
            self._basis = np.delete(self._basis, i, axis=0)
            del self._labels[i]
            del self._plot_labels[i]
            del self._interaction_qubits[i]
            del self._interaction_labels[i]
        self._lie_algebra_dim = self._basis.shape[0]
        return True
    
    @property
    def basis(self):
        return self._basis

    @property
    def labels(self):
        return self._labels if self._labels else None 

    @property
    def plot_labels(self):
        return self._plot_labels

    @property
    def interaction_labels(self):
        return self._interaction_labels

    @property
    def interaction_qubits(self):
        return self._interaction_qubits

    @property
    def interaction_graph(self):
        return self._interaction_graph

    @property
    def interaction_map(self):
        return self._interaction_map
    
    @property
    def n(self):
        return self._n

    @property
    def local_dim(self):
        return self._local_dim
    
    @property
    def dim(self):
        return self._dim

    @property
    def lie_algebra_dim(self):
        return self._lie_algebra_dim
    
    @property
    def shape(self):
        return self._basis.shape
    
    def __len__(self):
        return self._basis.shape[0]

    def generate_parameter_list(self, parameter_map: dict):
        """
        Generates a list of parameters for the Hamiltonian based on the parameter map.

        Parameters
        ----------
        parameter_map : dict
            A dictionary where the keys are the qubits and the values are dictionaries of interactions with parameter values.

        Returns
        -------
        list
            A parameter list.
        """
        parameter_list = []
        for label in self.labels:
            new_label = ""
            qubits = []
            for i, c in enumerate(label):
                if c == "I":
                    new_label += ""
                else:
                    new_label += f"{c}".lower()
                    qubits.append(i+1)
            qubits = tuple(qubits) if len(qubits) > 1 else qubits[0]

            interactions = parameter_map.get(qubits)
            if interactions is not None: 
                param = interactions.get(new_label, 0)
                parameter_list.append(param)
            else:
                parameter_list.append(0)
        return parameter_list


    def generate_bounds(self, bounds_map: dict, piecewise_steps: int):
        """
        Generates a list of parameter bounds for the Hamiltonian based on the boundary map.

        Parameters
        ----------
        boundary_map : dict
            A dictionary where the keys are label strings and the values are dictionaries of interactions with (min, max) tuples.
        parameters : list
            A list of the current parameter values.

        Returns
        -------
        list
            A tuple of a min list and a max list corresponding to each parameter.
        """
        # piecewise_steps = parameters.shape[0]
        upper_bounds = [[] for _ in range(piecewise_steps)]
        lower_bounds = [[] for _ in range(piecewise_steps)]
        for label in self.labels:
            new_label = ""
            qubits = []
            for i, c in enumerate(label):
                if c == "I":
                    new_label += ""
                else:
                    new_label += f"{c}".lower()
                    qubits.append(i+1)
            qubits = tuple(qubits) if len(qubits) > 1 else qubits[0]

            bounds = bounds_map.get(new_label)
            for gate in range(piecewise_steps):
                if bounds is not None: 
                    index = len(lower_bounds[gate])
                    # lower_bounds[gate].append(bounds[0] - parameters[gate, index])
                    # upper_bounds[gate].append(bounds[1] - parameters[gate, index])
                    lower_bounds[gate].append(bounds[0])
                    upper_bounds[gate].append(bounds[1])
                else:
                    lower_bounds[gate].append(-jnp.inf)
                    upper_bounds[gate].append(jnp.inf)
        return lower_bounds, upper_bounds


class Hamiltonian:
    """
    Object representation of the Hamiltonians defined by a Lie algebra basis and phi components.

    Parameters
    ----------
    basis : gnd.Basis
        Generally the PauliBasis object, which defines the basis of the Lie algebra for the phi 
        parameters.
    parameters : np.array
        Numpy array of same length as basis defining the components of the basis.

    Attributes
    ----------
    basis : gnd.PauliBasis
        Basis object including the basis elements and labels.
    parameters : np.array
        Lie algebra component vector.
    matrix : np.ndarray
        Matrix representation of the Hamiltonian.
    n : int 
        The number of qubits that the Hamiltonian acts on is stored.
    """

    def __init__(self, basis, parameters):
        self.basis = basis
        self.parameters = parameters
        self.matrix = self._matrix()
        self.unitary = Unitary(spla.expm(1.j * self.matrix))

    def geodesic_hamiltonian(self, target_unitary):
        """
        Returns the geodesic to a target unitary.

        Parameters
        ----------
        target_unitary : np.ndarray
            The unitary target

        Returns
        -------
        Hamiltonian
            The geodesic hamiltonian.
        """
        g = -1.j * spla.logm(self.unitary.matrix.conj().T @ target_unitary)
        g_params = Hamiltonian.parameters_from_hamiltonian(g, self.basis)
        return Hamiltonian(self.basis, g_params)

    def fidelity(self, unitary_matrix):
        """
        Returns the fidelity to a target unitary.

        Parameters
        ----------
        unitary : np.ndarray
            The unitary target

        Returns
        -------
        float
            The fidelity.
        """
        return Unitary.unitary_fidelity(self.unitary.matrix, unitary_matrix)

    @staticmethod
    def parameters_from_hamiltonian(hamiltonian, basis):
        """
        Returns the parameters from a Hamiltonian.

        Parameters
        ----------
        hamiltonian : np.ndarray
            The Hamiltonian for which we want to find the parameters.
        basis: gnd.Basis
            The basis to find the parameters in.

        Returns
        -------
        np.array
            Parameters vector.
        """
        return np.real(np.einsum("ijk, kj->i", basis.basis, hamiltonian)) / (len(hamiltonian[0]))

    def _matrix(self):
        return np.einsum("ijk,i->jk", self.basis.basis, self.parameters)


class Unitary:
    """
    Object representation of the Unitary defined by a matrix.

    Parameters
    ----------
    unitary : np.ndarray
        Matrix of size 2^n x 2^n that describes a unitary acting on n qubits

    Attributes
    ----------
    matrix : np.ndarray
        Matrix representation of the unitary.
    n : int 
        The number of qubits that the unitary acts on.
    """

    def __init__(self, unitary_matrix):
        self.matrix, self.n = Unitary._check_is_unitary(unitary_matrix)

    def parameters(self, basis):
        """
        Calculate parameters from the unitary using the principal logarithm.

        Parameters
        ----------

        Returns
        -------

        """
        return Unitary.parameters_from_unitary(self.matrix, basis)

    def fidelity(self, unitary_matrix):
        """
        Returns the fidelity to a target unitary.

        Parameters
        ----------
        unitary : np.ndarray
            The unitary target

        Returns
        -------
        float
            The fidelity.
        """
        return Unitary.unitary_fidelity(self.matrix, unitary_matrix)

    def geodesic_hamiltonian(self, basis, target_unitary):
        """
        Returns the geodesic to a target unitary.

        Parameters
        ----------
        target_unitary : np.ndarray
            The unitary target

        Returns
        -------
        Unitary
            The geodesic unitary.
        """
        g = -1.j * spla.logm(self.matrix.conj().T @ target_unitary)
        params = Hamiltonian.parameters_from_hamiltonian(g, basis)
        return Hamiltonian(basis, params)

    def __matmul__(self, other):
        return Unitary(self.matrix @ other.matrix)

    @staticmethod
    def _check_is_unitary(unitary_matrix):
        if not np.allclose(np.eye(len(unitary_matrix)), unitary_matrix @ unitary_matrix.T.conj()):
            raise ValueError("Matrix given to Unitary must be unitary: U U^dagger = U^dagger U = I")
        if not unitary_matrix.shape[0] == unitary_matrix.shape[1]:
            raise ValueError("Matrix must be square")
        return unitary_matrix, int(np.log2(len(unitary_matrix)))

    @staticmethod
    @jax.jit
    def unitary_fidelity(matrixA, matrixB):
        return jnp.abs(jnp.trace(matrixA.conj().T @ matrixB)) / len(matrixA[0])

    @staticmethod
    def parameters_from_unitary(unitary_matrix, basis):
        """
        Returns the parameters from a unitary.

        Parameters
        ----------
        unitary : np.ndarray
            The unitary for which we want to find the parameters.
        basis: gnd.Basis
            The basis to find the parameters in.

        Returns
        -------
        np.array
            Parameters vector.
        """
        return Hamiltonian.parameters_from_hamiltonian(-1.j * spla.logm(unitary_matrix), basis)