//
// Created by Florent Delgrange on 2019-02-12.
//

#include "TotalPayoffGame.h"

namespace sw {
    namespace game {

        template<typename ValueType>
        TotalPayoffGame<ValueType>::TotalPayoffGame(
                const storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                const std::string &rewardModelName,
                const storm::storage::BitVector &restrictedStateSpace,
                const storm::storage::BitVector &enabledActions)
                : MdpGame<ValueType>(mdp, restrictedStateSpace, enabledActions),
                  rewardModelName(rewardModelName),
                  rewardModel(mdp.getRewardModel(rewardModelName)) {}

        template<typename ValueType>
        std::vector<ValueType> TotalPayoffGame<ValueType>::maxTotalPayoffInf() const {
            std::vector<ValueType> const& weights = rewardModel.getStateActionRewardVector();
            std::vector<ValueType> absoluteWeights(weights.size());
            std::transform(weights.begin(), weights.end(), absoluteWeights.begin(),
                           [](ValueType w) -> ValueType { return storm::utility::abs(w); });
            return maxTotalPayoffInf(
                    storm::Environment(),
                    this->restrictedStateSpace,
                    this->enabledActions,
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new successorsP1(state, this->matrix, this->enabledActions) ); },
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new successorsP2(state, this->matrix, this->enabledActions) ); },
                    [&](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return weights[s_prime]; },
                    [](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return storm::utility::zero<ValueType>(); },
                    storm::utility::maximum(absoluteWeights))
                    .max;
        }

        template<typename ValueType>
        std::vector<ValueType> TotalPayoffGame<ValueType>::minTotalPayoffSup() const {
            std::vector<ValueType> const& weights = rewardModel.getStateActionRewardVector();
            std::vector<ValueType> oppositeWeights(weights.size());
            std::transform(weights.begin(), weights.end(), oppositeWeights.begin(),
                           [](ValueType w) -> ValueType { return w * -1; });
            std::vector<ValueType> absoluteWeights(weights.size());
            std::transform(weights.begin(), weights.end(), absoluteWeights.begin(),
                           [](ValueType w) -> ValueType { return storm::utility::abs(w); });
            std::vector<ValueType> result = maxTotalPayoffInf(
                    storm::Environment(),
                    this->enabledActions,
                    this->restrictedStateSpace,
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new successorsP2(state, this->matrix, this->enabledActions) ); },
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new successorsP1(state, this->matrix, this->enabledActions) ); },
                    [](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return storm::utility::zero<ValueType>(); },
                    [&](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return oppositeWeights[s_prime]; },
                    storm::utility::maximum(absoluteWeights))
                    .min;
            std::transform(result.begin(), result.end(), result.begin(), [](ValueType v) -> ValueType { return v * -1; });
            return result;
        }

        template <typename ValueType>
        bool TotalPayoffGame<ValueType>::valuesEqual(
                typename TotalPayoffGame<ValueType>::Values const& x,
                typename TotalPayoffGame<ValueType>::Values const& y,
                ValueType precision,
                bool relativeError) const {

            if (not relativeError) {
                precision *= storm::utility::convertNumber<ValueType>(2.0);
            }
            return storm::utility::vector::equalModuloPrecision<ValueType>(x.max, y.max, precision, relativeError) and
                   storm::utility::vector::equalModuloPrecision<ValueType>(x.min, y.min, precision, relativeError);
        }

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::Values TotalPayoffGame<ValueType>::maxTotalPayoffInf(
                storm::Environment const& env,
                storm::storage::BitVector const& maximizerStateSpace,
                storm::storage::BitVector const& minimizerStateSpace,
                std::function<std::unique_ptr<successors>(uint_fast64_t)> const& maximizerSuccessors,
                std::function<std::unique_ptr<successors>(uint_fast64_t)> const& minimizerSuccessors,
                std::function<ValueType(uint_fast64_t, uint_fast64_t)> const& wMaxToMin,
                std::function<ValueType(uint_fast64_t, uint_fast64_t)> const& wMinToMax,
                ValueType W) const {

            using Values = TotalPayoffGame<ValueType>::Values;
            std::shared_ptr<Values> Y = std::shared_ptr<Values>(new Values()), Y_pre, X, X_pre;
            Y->max = std::vector<ValueType>(maximizerStateSpace.size(), -1 * storm::utility::infinity<ValueType>());
            Y->min = std::vector<ValueType>(minimizerStateSpace.size(), -1 * storm::utility::infinity<ValueType>());
            ValueType valuesUpperBound = (maximizerStateSpace.size() + minimizerStateSpace.size() - 1) * W;
            ValueType valuesLowerBound = - 1 * valuesUpperBound;
            // for the vector equality check
            auto precision = storm::utility::convertNumber<ValueType>(env.solver().minMax().getPrecision());
            bool relative = env.solver().minMax().getRelativeTerminationCriterion();

            uint_fast64_t external_counter = 0;
            uint_fast64_t internal_counter = 0;
            uint_fast64_t max_iter = maximizerStateSpace.size() + minimizerStateSpace.size() *
                    (2 * (maximizerStateSpace.size() + minimizerStateSpace.size() - 1) * storm::utility::convertNumber<uint_fast64_t>(W) + 1);

            do {
                ++external_counter;
                // incorporate weights of the previous copy of the game into the current copy of the game
                Y_pre = Y;
                Y = std::shared_ptr<Values>(new Values());

                Y->max = std::vector<ValueType>(maximizerStateSpace.size());
                for (uint_fast64_t s: maximizerStateSpace) {
                    Y->max[s] = storm::utility::max<ValueType>(storm::utility::zero<ValueType>(), Y_pre->max[s]);
                }

                Y->min = std::vector<ValueType>(minimizerStateSpace.size());
                for (uint_fast64_t s: minimizerStateSpace) {
                    Y->min[s] = storm::utility::max<ValueType>(storm::utility::zero<ValueType>(), Y_pre->min[s]);
                }

                X = std::shared_ptr<Values>(new Values());
                X->max = std::vector<ValueType>(maximizerStateSpace.size(), storm::utility::infinity<ValueType>());
                X->min = std::vector<ValueType>(minimizerStateSpace.size(), storm::utility::infinity<ValueType>());

                // min-cost reachability
                do {
                    ++ internal_counter;
                    X_pre = X;
                    X = std::shared_ptr<Values>(new Values());

                    // maximizer phase
                    X->max = std::vector<ValueType>(maximizerStateSpace.size());
                    for (uint_fast64_t s: maximizerStateSpace) {
                        X->max[s] = -1 * storm::utility::infinity<ValueType>();
                        for (uint_fast64_t s_prime: *maximizerSuccessors(s)) {
                            ValueType x = wMaxToMin(s, s_prime) + storm::utility::min(X_pre->min[s_prime], Y->min[s_prime]);
                            if (x > X->max[s]) {
                                X->max[s] = x;
                            }
                        }
                    }
                    // minimizer phase
                    X->min = std::vector<ValueType>(minimizerStateSpace.size());
                    for (uint_fast64_t s: minimizerStateSpace) {
                        X->min[s] = storm::utility::infinity<ValueType>();
                        for (uint_fast64_t s_prime: *minimizerSuccessors(s)) {
                            ValueType x = wMinToMax(s, s_prime) + storm::utility::min(X_pre->max[s_prime], Y->max[s_prime]);
                            if (x < X->min[s]) {
                                X->min[s] = x;
                            }
                        }
                    }
                    // lower bound checking phase
                    for (uint_fast64_t s: maximizerStateSpace) {
                        if (X->max[s] < valuesLowerBound) {
                            X->max[s] = -1 * storm::utility::infinity<ValueType>();
                        }
                    }
                    for (uint_fast64_t s: minimizerStateSpace) {
                        if (X->min[s] < valuesLowerBound) {
                            X->min[s] = -1 * storm::utility::infinity<ValueType>();
                        }
                    }
                } while (not valuesEqual(*X, *X_pre, precision, relative));

                Y = X;
                // upper bound checking phase
                for (uint_fast64_t s: maximizerStateSpace) {
                    if (Y->max[s] > valuesUpperBound) {
                        Y->max[s] = storm::utility::infinity<ValueType>();
                    }
                }
                for (uint_fast64_t s: minimizerStateSpace) {
                    if (Y->min[s] > valuesUpperBound) {
                        Y->min[s] = storm::utility::infinity<ValueType>();
                    }
                }
            } while (not valuesEqual(*Y, *Y_pre, precision, relative) or external_counter >= max_iter);
            return *Y;
        }


        /*!
         * Iterators implementation
         */

        template <typename ValueType>
        TotalPayoffGame<ValueType>::successors::successors(
                uint_fast64_t state,
                const storm::storage::SparseMatrix<ValueType> &matrix,
                const storm::storage::BitVector &enabledActions)
                : state(state), matrix(matrix), enabledActions(enabledActions) {}

        template <typename ValueType>
        TotalPayoffGame<ValueType>::successors::~successors() = default;

        template <typename ValueType>
        TotalPayoffGame<ValueType>::successorsP1::successorsP1(
                uint_fast64_t state,
                const storm::storage::SparseMatrix<ValueType> &matrix,
                storm::storage::BitVector const& enabledActions)
                : successors(state, matrix, enabledActions) {}

        template <typename ValueType>
        TotalPayoffGame<ValueType>::successorsP2::successorsP2(
                uint_fast64_t action,
                const storm::storage::SparseMatrix<ValueType> &matrix,
                storm::storage::BitVector const& enabledActions)
                : successors(action, matrix, enabledActions) {}

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator TotalPayoffGame<ValueType>::successors::end() {
            return iterator(0, 0, this->enabledActions);
        }

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator TotalPayoffGame<ValueType>::successorsP1::begin() {
            uint_fast64_t end = this->state == this->matrix.getRowGroupCount() - 1 ?
                                this->matrix.getRowCount()
                                : this->matrix.getRowGroupIndices()[this->state + 1];
            return iterator(this->matrix.getRowGroupIndices()[this->state], end, this->enabledActions);
        }

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator TotalPayoffGame<ValueType>::successorsP2::begin() {
            return this->enabledActions[this->state] ?
                   iterator(this->matrix.getRow(this->state).begin(), this->matrix.getRow(this->state).end(), this->enabledActions)
                   : iterator(0, 0, this->enabledActions);
        }

        template<typename ValueType>
        TotalPayoffGame<ValueType>::iterator::iterator(uint_fast64_t rowBegin, uint_fast64_t rowEnd,
                storm::storage::BitVector const& enabledActions)
                : rowEnd(rowEnd), iterate_on_columns(false), enabledActions(enabledActions) {
                    this->currentRow = this->enabledActions.getNextSetIndex(rowBegin);
        }

        template<typename ValueType>
        TotalPayoffGame<ValueType>::iterator::iterator(
                typename std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>::const_iterator entriesIteratorBegin,
                typename std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>::const_iterator entriesIteratorEnd,
                storm::storage::BitVector const& enabledActions)
                : currentRow(0), rowEnd(0),
                  matrixEntryIterator(entriesIteratorBegin),
                  ptr_end(entriesIteratorEnd),
                  iterate_on_columns(true),
                  enabledActions(enabledActions) {
            this->currentColumn = this->matrixEntryIterator->getColumn();
            this->matrixEntryIterator++;
        }

        template<typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator& TotalPayoffGame<ValueType>::iterator::operator++() {
            if (this->matrixEntryIterator == this->ptr_end) {
                this->iterate_on_columns = false;
            }
            if (this->currentRow < this->rowEnd) {
                this->currentRow = this->enabledActions.getNextSetIndex(currentRow + 1);
            }
            else if (this->iterate_on_columns) {
                this->currentColumn = this->matrixEntryIterator->getColumn();
                this->matrixEntryIterator++;
            }
            return *this;
        }

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator TotalPayoffGame<ValueType>::iterator::operator++(int) {
            iterator it = *this;
            ++*this;
            return it;
        }

        template <typename ValueType>
        bool TotalPayoffGame<ValueType>::iterator::operator!=(const TotalPayoffGame<ValueType>::iterator &otherIterator) {
            bool stopped = this->currentRow >= this->rowEnd and otherIterator.currentRow >= otherIterator.rowEnd
                         and (not this->iterate_on_columns) and (not otherIterator.iterate_on_columns);
            bool sameCurrentRow = not this->iterate_on_columns and not otherIterator.iterate_on_columns
                            and this->currentRow == otherIterator.currentRow and this->rowEnd == otherIterator.rowEnd;
            bool sameCurrentColumn = this->iterate_on_columns and otherIterator.iterate_on_columns
                            and this->currentColumn == otherIterator.currentColumn and this->ptr_end == otherIterator.ptr_end;
            bool isEqual = stopped or sameCurrentRow or sameCurrentColumn;
            return not isEqual;
        }

        template <typename ValueType>
        uint_fast64_t TotalPayoffGame<ValueType>::iterator::operator*() {
            if (not iterate_on_columns) {
                return this->currentRow;
            }
            else {
                return this->currentColumn;
            }
        }

        template class TotalPayoffGame<double>;
        template class TotalPayoffGame<storm::RationalNumber>;

    }
}
