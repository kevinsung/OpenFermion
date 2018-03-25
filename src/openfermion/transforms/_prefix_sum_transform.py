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


def prefix_sum_transform(operator, n_qubits,
                         occupation_set, parity_set, update_set):
    """Apply a transform specified by a prefix sum algorithm.

    Args:
        operator (openfermion.ops.FermionOperator):
            A FermionOperator to transform.
        n_qubits (int|None):
            Can force the number of qubits in the resulting operator above the
            number that appear in the input operator.
        occupation_set(callable):
        parity_set(callable):
        update_set(callable):

    Returns:
        transformed_operator: An instance of the QubitOperator class.

    Raises:
        ValueError: Invalid number of qubits specified.
    """
    transformed_terms = (
        _transform_operator_term(term, operator.terms[term],
                                 occupation_set, parity_set, update_set)
        for term in operator.terms
    )
    return inline_sum(summands=transformed_terms, seed=QubitOperator())


def _transform_operator_term(term, coefficient,
                             occupation_set, parity_set, update_set):
    """
    Args:
        term (list[tuple[int, int]]):
            A list of (mode, raising-vs-lowering) ladder operator terms.
        coefficient (float):
        occupation_set(callable):
        parity_set(callable):
        update_set(callable):
    Returns:
        QubitOperator:
    """

    # Build the Bravyi-Kitaev transformed operators.
    transformed_ladder_ops = (
        _transform_ladder_operator(ladder_operator,
                                   occupation_set, parity_set, update_set)
        for ladder_operator in term
    )
    return inline_product(factors=transformed_ladder_ops,
                          seed=QubitOperator((), coefficient))
                          


def _transform_ladder_operator(ladder_operator,
                               occupation_set, parity_set, update_set):
    """
    Args:
        ladder_operator (tuple[int, int]): the ladder operator
        occupation_set(callable):
        parity_set(callable):
        update_set(callable):
    Returns:
        QubitOperator
    """
    index, action = ladder_operator

    update_set_ = update_set(index)
    occupation_set_ = occupation_set(index)
    if index == 0:
        parity_set_ = set()
    else:
        parity_set_ = parity_set(index - 1)

    # Initialize the transformed majorana operator (a_p^\dagger + a_p) / 2
    transformed_operator = QubitOperator(
            [(i, 'X') for i in update_set_] +
            [(i, 'Z') for i in parity_set_],
            .5)
    # Get the transformed (a_p^\dagger - a_p) / 2
    # Below is equivalent to X(update_set) * Z(parity_set ^ occupation_set)
    transformed_majorana_difference = QubitOperator(
            [(index, 'Y')] +
            [(i, 'X') for i in update_set_ - {index}] +
            [(i, 'Z') for i in (parity_set_ ^ occupation_set_) - {index}],
            -.5j)

    # Raising
    if action == 1:
        transformed_operator += transformed_majorana_difference
    # Lowering
    else:
        transformed_operator -= transformed_majorana_difference

    return transformed_operator


def inline_sum(summands, seed):
    """Computes a sum, using the __iadd__ operator.
    Args:
        seed (T): The starting total. The zero value.
        summands (iterable[T]): Values to add (with +=) into the total.
    Returns:
        T: The result of adding all the factors into the zero value.
    """
    for r in summands:
        seed += r
    return seed


def inline_product(factors, seed):
    """Computes a product, using the __imul__ operator.
    Args:
        seed (T): The starting total. The unit value.
        factors (iterable[T]): Values to multiply (with *=) into the total.
    Returns:
        T: The result of multiplying all the factors into the unit value.
    """
    for r in factors:
        seed *= r
    return seed
