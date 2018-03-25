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
from openfermion.transforms import (prefix_sum_transform,
                                    occupation_set_parity,
                                    parity_set_parity,
                                    update_set_parity)
from openfermion.utils import count_qubits


def parity_transform(operator, n_qubits=None):
    """Apply the parity transform.

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

    def update_set(index):
        return update_set_parity(index, n_qubits)

    return prefix_sum_transform(operator, n_qubits,
                                occupation_set_parity,
                                parity_set_parity,
                                update_set)
