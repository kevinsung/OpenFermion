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

from openfermion.ops import QubitOperator
from openfermion.utils import count_qubits


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


def update_set_parity(index, n_qubits):
    """The bits that need to be updated upon flipping the occupancy
    of a mode."""
    return range(index, n_qubits)


def occupation_set_parity(index):
    """The bits whose parity stores the occupation of mode `index`."""
    if index == 0:
        return {index}
    else:
        return {index, index + 1}


def parity_set_parity(index):
    """The bits whose parity stores the parity of the bits 0 .. `index`."""
    return {index}
