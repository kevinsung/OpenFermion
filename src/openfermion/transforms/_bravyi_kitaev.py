#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Bravyi-Kitaev transform on fermionic operators."""

from openfermion.transforms._prefix_sum_transform import prefix_sum_transform
from openfermion.utils import count_qubits


def bravyi_kitaev(operator, n_qubits=None):
    """Apply the Bravyi-Kitaev transform.

    Implementation from arXiv:quant-ph/0003137 and
    "A New Data Structure for Cumulative Frequency Tables" by Peter M. Fenwick.

    Note that this implementation is equivalent to the one described in
    arXiv:1208.5986, and is different from the one described in
    arXiv:1701.07072. The one described in arXiv:1701.07072 is implemented
    in OpenFermion as `bravyi_kitaev_tree`.

    Args:
        operator (openfermion.ops.FermionOperator):
            A FermionOperator to transform.
        n_qubits (int|None):
            Can force the number of qubits in the resulting operator above the
            number that appear in the input operator.

    Returns:
        transformed_operator: An instance of the QubitOperator class.

    Raises:
        ValueError: Invalid number of qubits specified.
    """
    # Compute the number of qubits.
    if n_qubits is None:
        n_qubits = count_qubits(operator)
    if n_qubits < count_qubits(operator):
        raise ValueError('Invalid number of qubits specified.')

    # Compute transformed operator.
    def update_set(index):
        return update_set_bravyi_kitaev(index, n_qubits)

    return prefix_sum_transform(operator, n_qubits,
                                occupation_set_bravyi_kitaev,
                                parity_set_bravyi_kitaev,
                                update_set)


def update_set_bravyi_kitaev(index, n_qubits):
    """The bits that need to be updated upon flipping the occupancy
    of a mode."""
    indices = set()

    # For bit manipulation we need to count from 1 rather than 0
    index += 1

    while index <= n_qubits:
        indices.add(index - 1)
        # Add least significant one to index
        # E.g. 00010100 -> 00011000
        index += index & -index
    return indices


def occupation_set_bravyi_kitaev(index):
    """The bits whose parity stores the occupation of mode `index`."""
    indices = set()

    # For bit manipulation we need to count from 1 rather than 0
    index += 1

    indices.add(index - 1)
    parent = index & (index - 1)
    index -= 1
    while index != parent:
        indices.add(index - 1)
        # Remove least significant one from index
        # E.g. 00010100 -> 00010000
        index &= index - 1
    return indices


def parity_set_bravyi_kitaev(index):
    """The bits whose parity stores the parity of the bits 0 .. `index`."""
    indices = set()

    # For bit manipulation we need to count from 1 rather than 0
    index += 1

    while index > 0:
        indices.add(index - 1)
        # Remove least significant one from index
        # E.g. 00010100 -> 00010000
        index &= index - 1
    return indices
