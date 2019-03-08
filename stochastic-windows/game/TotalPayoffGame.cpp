//
// Created by Florent Delgrange on 2019-02-12.
//

#include "TotalPayoffGame.h"
#include <time.h>

namespace sw {
    namespace game {

        template<typename ValueType>
        TotalPayoffGame<ValueType>::TotalPayoffGame(
                const storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                const std::string &rewardModelName,
                const storm::storage::BitVector &restrictedStateSpace,
                const storm::storage::BitVector &enabledActions,
                bool initTransitionStructure)
                : MdpGame<ValueType>(mdp, restrictedStateSpace, enabledActions),
                  rewardModelName(rewardModelName),
                  rewardModel(mdp.getRewardModel(rewardModelName)) {

            if (initTransitionStructure) {
                this->initBackwardTransitions(this->backwardTransitions);
                this->forwardTransitions.successors = std::vector<std::forward_list<uint_fast64_t>>(this->matrix.getRowCount());
                for (uint_fast64_t const& state: this->restrictedStateSpace) {
                    for (uint_fast64_t action = this->enabledActions.getNextSetIndex(this->matrix.getRowGroupIndices()[state]);
                         action < this->matrix.getRowGroupIndices()[state + 1];
                         action = this->enabledActions.getNextSetIndex(action + 1)) {
                        for (const auto &entry: this->matrix.getRow(action)) {
                            const uint_fast64_t& successorState = entry.getColumn();
                            // in total-payoff games, actions may lead to states not belonging to the restricted state space
                            if (this->restrictedStateSpace[successorState]) {
                                this->forwardTransitions.successors[action].push_front(successorState);
                            }
                        }
                    }
                }
            }
        }

        template<typename ValueType>
        std::vector<ValueType> TotalPayoffGame<ValueType>::maxTotalPayoffInf() const {
            std::vector<ValueType> const& weights = rewardModel.getStateActionRewardVector();
            std::vector<ValueType> absoluteWeights(weights.size());
            std::transform(weights.begin(), weights.end(), absoluteWeights.begin(),
                           [](ValueType w) -> ValueType { return storm::utility::abs(w); });

            std::function<std::unique_ptr<successors>(uint_fast64_t)> p2TransitionFunction;
            if (this->forwardTransitions.successors.empty()) {
                p2TransitionFunction = [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                    return std::unique_ptr<successors>( new successorsP2(state, this->matrix, this->restrictedStateSpace, this->enabledActions)); };
            }
            else {
                p2TransitionFunction = [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                    return std::unique_ptr<successors>( new forwardSuccessorsP2(state, this->matrix, this->forwardTransitions, this->restrictedStateSpace, this->enabledActions) ); };
            }

            return maxTotalPayoffInf(
                    storm::Environment(),
                    this->restrictedStateSpace,
                    this->enabledActions,
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new successorsP1(state, this->matrix, this->restrictedStateSpace, this->enabledActions) ); },
                    p2TransitionFunction,
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

            std::function<std::unique_ptr<successors>(uint_fast64_t)> p2TransitionFunction;
            if (this->forwardTransitions.successors.empty()) {
                p2TransitionFunction = [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                    return std::unique_ptr<successors>( new successorsP2(state, this->matrix, this->restrictedStateSpace, this->enabledActions)); };
            }
            else {
                p2TransitionFunction = [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                    return std::unique_ptr<successors>( new forwardSuccessorsP2(state, this->matrix, this->forwardTransitions, this->restrictedStateSpace, this->enabledActions)); };
            }

            std::vector<ValueType> result = maxTotalPayoffInf(
                    storm::Environment(),
                    this->enabledActions,
                    this->restrictedStateSpace,
                    p2TransitionFunction,
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new successorsP1(state, this->matrix, this->restrictedStateSpace, this->enabledActions) ); },
                    [](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return storm::utility::zero<ValueType>(); },
                    [&](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return oppositeWeights[s_prime]; },
                    storm::utility::maximum(absoluteWeights))
                    .min;
            std::transform(result.begin(), result.end(), result.begin(), [](ValueType v) -> ValueType { return v * -1; });
            return result;
        }

        template <typename ValueType>
        GameStates TotalPayoffGame<ValueType>::negSupTP() const {
            std::vector<ValueType> const& weights = rewardModel.getStateActionRewardVector();
            std::vector<ValueType> oppositeWeights(weights.size());
            std::transform(weights.begin(), weights.end(), oppositeWeights.begin(),
                           [](ValueType w) -> ValueType { return w * -1; });
            std::vector<ValueType> absoluteWeights(weights.size());
            std::transform(weights.begin(), weights.end(), absoluteWeights.begin(),
                           [](ValueType w) -> ValueType { return storm::utility::abs(w); });

            std::function<std::unique_ptr<successors>(uint_fast64_t)> p2TransitionFunction;
            if (this->forwardTransitions.successors.empty()) {
                p2TransitionFunction = [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                    return std::unique_ptr<successors>( new successorsP2(state, this->matrix, this->restrictedStateSpace, this->enabledActions)); };
            }
            else {
                p2TransitionFunction = [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                    return std::unique_ptr<successors>( new forwardSuccessorsP2(state, this->matrix, this->forwardTransitions, this->restrictedStateSpace, this->enabledActions)); };
            }

            Values result = maxTotalPayoffInf(
                    storm::Environment(),
                    this->enabledActions,
                    this->restrictedStateSpace,
                    p2TransitionFunction,
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new successorsP1(state, this->matrix, this->restrictedStateSpace, this->enabledActions) ); },
                    [](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return storm::utility::zero<ValueType>(); },
                    [&](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return oppositeWeights[s_prime]; },
                    storm::utility::maximum(absoluteWeights));
            GameStates badStates;
            badStates.p1States = storm::storage::BitVector(this->restrictedStateSpace.size(), false);
            badStates.p2States = storm::storage::BitVector(this->enabledActions.size(), false);
            for (uint_fast64_t s: this->restrictedStateSpace) {
                if (result.min[s] > 0) {
                    badStates.p1States.set(s, true);
                }
            }
            for (uint_fast64_t s: this->enabledActions)  {
                if (result.max[s] > 0) {
                    badStates.p2States.set(s, true);
                }
            }
            return badStates;
        }

        template <typename ValueType>
        bool TotalPayoffGame<ValueType>::equalModuloPrecision(const ValueType &val1, const ValueType &val2,
                                                              const ValueType &precision, bool relativeError) const {

            // infinity handling
            if (val1 == storm::utility::infinity<ValueType>() or val2 == storm::utility::infinity<ValueType>()) {
                return val1 == val2;
            }

            if (val1 == -1 * storm::utility::infinity<ValueType>() or val2 == -1 * storm::utility::infinity<ValueType>()) {
                return val1 == val2;
            }

            if (relativeError) {
                if (storm::utility::isZero<ValueType>(val1)) {
                    return storm::utility::isZero(val2);
                }
                ValueType relDiff = (val1 - val2)/val1;
                if (storm::utility::abs(relDiff) > precision) {
                    return false;
                }
            } else {
                ValueType diff = val1 - val2;
                if (storm::utility::abs(diff) > precision) {
                    return false;
                }
            }
            return true;
        }

        // Specialization for double as the relative check for doubles very close to zero is not meaningful.
        template<>
        inline bool TotalPayoffGame<double>::equalModuloPrecision(double const& val1, double const& val2,
                                                                  double const& precision, bool relativeError) const {

            // infinity handling
            if (val1 == storm::utility::infinity<double>() or val2 == storm::utility::infinity<double>()) {
                return val1 == val2;
            }

            if (val1 == -1 * storm::utility::infinity<double>() or val2 == -1 * storm::utility::infinity<double>()) {
                return val1 == val2;
            }

            if (relativeError) {
                if (storm::utility::isAlmostZero(val2)) {
                    return storm::utility::isAlmostZero(val1);
                }
                double relDiff = (val1 - val2)/val1;
                if (storm::utility::abs(relDiff) > precision) {
                    return false;
                }
            } else {
                double diff = val1 - val2;
                if (storm::utility::abs(diff) > precision) {
                    return false;
                }
            }
            return true;
        }

        template<typename ValueType>
        bool TotalPayoffGame<ValueType>::vectorEquality(const std::vector<ValueType> &vectorLeft,
                                                        const std::vector<ValueType> &vectorRight,
                                                        const ValueType &precision, bool relativeError) const {

            STORM_LOG_ASSERT(vectorLeft.size() == vectorRight.size(), "Lengths of vectors does not match.");

            auto leftIt = vectorLeft.begin();
            auto leftIte = vectorLeft.end();
            auto rightIt = vectorRight.begin();
            for (; leftIt != leftIte; ++leftIt, ++rightIt) {
                if (!this->equalModuloPrecision(*leftIt, *rightIt, precision, relativeError)) {
                    return false;
                }
            }

            return true;
        }


        template <typename ValueType>
        bool TotalPayoffGame<ValueType>::valuesEqual(
                typename TotalPayoffGame<ValueType>::Values const& X,
                typename TotalPayoffGame<ValueType>::Values const& Y,
                ValueType precision,
                bool relativeError) const {

            if (not relativeError) {
                precision *= storm::utility::convertNumber<ValueType>(2.0);
            }

            return this->vectorEquality(X.max, Y.max, precision, relativeError) and
                   this->vectorEquality(X.min, Y.min, precision, relativeError);
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
            //ValueType upperBound = (maximizerStateSpace.size() + minimizerStateSpace.size() - 1) * W;
            ValueType upperBound = storm::utility::convertNumber<ValueType>(this->restrictedStateSpace.getNumberOfSetBits() - 1) * W;
            ValueType lowerBound = -1 * upperBound;
            // for the vector equality check
            auto precision = storm::utility::convertNumber<ValueType>(env.solver().minMax().getPrecision());
            bool relative = env.solver().minMax().getRelativeTerminationCriterion();

            // uint_fast64_t external_counter = 0;
            // uint_fast64_t internal_counter = 0;
            // clock_t start = clock();

            do {
                // ++external_counter;
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
                    // ++ internal_counter;
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
                        if (X->max[s] < lowerBound) {
                            X->max[s] = -1 * storm::utility::infinity<ValueType>();
                        }
                    }
                    for (uint_fast64_t s: minimizerStateSpace) {
                        if (X->min[s] < lowerBound) {
                            X->min[s] = -1 * storm::utility::infinity<ValueType>();
                        }
                    }
                } while (not valuesEqual(*X, *X_pre, precision, relative));

                Y = X;
                // upper bound checking phase
                for (uint_fast64_t s: maximizerStateSpace) {
                    if (Y->max[s] > upperBound) {
                        Y->max[s] = storm::utility::infinity<ValueType>();
                    }
                }
                for (uint_fast64_t s: minimizerStateSpace) {
                    if (Y->min[s] > upperBound) {
                        Y->min[s] = storm::utility::infinity<ValueType>();
                    }
                }
            } while (not valuesEqual(*Y, *Y_pre, precision, relative));
            // clock_t stop = clock();
            // double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
            // printf("\nTime elapsed: %.5f | internal iterations: %llu | external iterations: %llu \n", elapsed, internal_counter, external_counter);
            return *Y;
        }

        template<typename ValueType>
        void TotalPayoffGame<ValueType>::initBackwardTransitions(BackwardTransitions &backwardTransitions) const {
            backwardTransitions.statePredecessors = std::vector<std::forward_list<uint_fast64_t>>(this->matrix.getRowGroupCount());
            backwardTransitions.actionPredecessor = std::vector<uint_fast64_t>(this->matrix.getRowCount());
            backwardTransitions.numberOfEnabledActions = std::vector<uint_fast64_t>(this->matrix.getRowGroupCount(), 0);
            for (uint_fast64_t const& state: this->restrictedStateSpace) {
                for (uint_fast64_t action = this->enabledActions.getNextSetIndex(this->matrix.getRowGroupIndices()[state]);
                     action < this->matrix.getRowGroupIndices()[state + 1];
                     action = this->enabledActions.getNextSetIndex(action + 1)) {
                    backwardTransitions.actionPredecessor[action] = state;
                    backwardTransitions.numberOfEnabledActions[state] += 1;
                    for (const auto &entry: this->matrix.getRow(action)) {
                        const uint_fast64_t& successorState = entry.getColumn();
                        // in total-payoff games, actions may lead to states not belonging to the restricted state space
                        if (this->restrictedStateSpace[successorState]) {
                            backwardTransitions.statePredecessors[successorState].push_front(action);
                        }
                    }
                }
            }
        }

        template<typename ValueType>
        BackwardTransitions const &TotalPayoffGame<ValueType>::getBackwardTransition() {
            return this->backwardTransitions;
        }


        /**
         * -------------------------------------------------------------------------------------------------------------
         * Iterators implementation
         * -------------------------------------------------------------------------------------------------------------
         */

        template <typename ValueType>
        TotalPayoffGame<ValueType>::successors::successors(
                uint_fast64_t state,
                const storm::storage::SparseMatrix<ValueType> &matrix,
                const storm::storage::BitVector &restrictedStateSpace,
                const storm::storage::BitVector &enabledActions)
                : state(state), matrix(matrix), restrictedStateSpace(restrictedStateSpace), enabledActions(enabledActions) {}

        template <typename ValueType>
        TotalPayoffGame<ValueType>::successors::~successors() = default;

        template <typename ValueType>
        TotalPayoffGame<ValueType>::successorsP1::successorsP1(
                uint_fast64_t state,
                const storm::storage::SparseMatrix<ValueType> &matrix,
                const storm::storage::BitVector &restrictedStateSpace,
                storm::storage::BitVector const& enabledActions)
                : successors(state, matrix, restrictedStateSpace, enabledActions) {}

        template <typename ValueType>
        TotalPayoffGame<ValueType>::successorsP2::successorsP2(
                uint_fast64_t action,
                const storm::storage::SparseMatrix<ValueType> &matrix,
                const storm::storage::BitVector &restrictedStateSpace,
                storm::storage::BitVector const& enabledActions)
                : successors(action, matrix, restrictedStateSpace, enabledActions) {}

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator TotalPayoffGame<ValueType>::successors::end() {
            return iterator(0, 0, this->restrictedStateSpace, this->enabledActions);
        }

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator TotalPayoffGame<ValueType>::successorsP1::begin() {
            uint_fast64_t end = this->state == this->matrix.getRowGroupCount() - 1 ?
                                this->matrix.getRowCount()
                                : this->matrix.getRowGroupIndices()[this->state + 1];
            return iterator(this->matrix.getRowGroupIndices()[this->state], end,
                            this->restrictedStateSpace, this->enabledActions);
        }

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator TotalPayoffGame<ValueType>::successorsP2::begin() {
            return this->enabledActions[this->state] ?
                   iterator(this->matrix.getRow(this->state).begin(), this->matrix.getRow(this->state).end(),
                            this->restrictedStateSpace, this->enabledActions)
                   : iterator(0, 0, this->restrictedStateSpace, this->enabledActions);
        }

        template<typename ValueType>
        TotalPayoffGame<ValueType>::iterator::iterator(uint_fast64_t rowBegin, uint_fast64_t rowEnd,
                storm::storage::BitVector const& restrictedStateSpace,
                storm::storage::BitVector const& enabledActions)
                : currentRow(enabledActions.getNextSetIndex(rowBegin)), rowEnd(rowEnd),
                  iterate_on_columns(false), forwardSuccessors(false),
                  restrictedStateSpace(restrictedStateSpace),
                  enabledActions(enabledActions) {}

        template<typename ValueType>
        TotalPayoffGame<ValueType>::iterator::iterator(
                typename std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>::const_iterator entriesIteratorBegin,
                typename std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>::const_iterator entriesIteratorEnd,
                storm::storage::BitVector const& restrictedStateSpace,
                storm::storage::BitVector const& enabledActions)
                : currentRow(0), rowEnd(0), currentColumn(entriesIteratorBegin->getColumn()),
                  matrixEntryIterator(entriesIteratorBegin), matrix_ptr_end(entriesIteratorEnd),
                  iterate_on_columns(true), forwardSuccessors(false),
                  restrictedStateSpace(restrictedStateSpace), enabledActions(enabledActions) {

            if (this->matrixEntryIterator != this->matrix_ptr_end) {
                ++this->matrixEntryIterator;
            }
            while (not this->restrictedStateSpace[this->currentColumn]) {
                this->currentColumn = this->matrixEntryIterator->getColumn();
                if (this->matrixEntryIterator != this->matrix_ptr_end) {
                    ++this->matrixEntryIterator;
                }
                // none of successors of the current action is in the restricted state space
                else {
                    this->iterate_on_columns = false;
                    break;
                }
            }

        }

        template<typename ValueType>
        TotalPayoffGame<ValueType>::iterator::iterator(
                typename std::forward_list<uint_fast64_t>::const_iterator successorsIteratorBegin,
                typename std::forward_list<uint_fast64_t>::const_iterator successorsIteratorEnd,
                storm::storage::BitVector const &restrictedStateSpace,
                storm::storage::BitVector const &enabledActions)
                : currentRow(0), rowEnd(0),
                successorsIterator(successorsIteratorBegin), successors_ptr_end(successorsIteratorEnd),
                iterate_on_columns(false), forwardSuccessors(true),
                restrictedStateSpace(restrictedStateSpace), enabledActions(enabledActions) {}

        template<typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator& TotalPayoffGame<ValueType>::iterator::operator++() {
            if (this->matrixEntryIterator == this->matrix_ptr_end) {
                this->iterate_on_columns = false;
            }
            if (this->currentRow < this->rowEnd) {
                this->currentRow = this->enabledActions.getNextSetIndex(currentRow + 1);
            }
            else if (this->iterate_on_columns) {
                this->currentColumn = this->matrixEntryIterator->getColumn();
                this->matrixEntryIterator++;
                if (not this->restrictedStateSpace[this->currentColumn]) {
                    ++*this;
                }
            }
            else if (this->forwardSuccessors) {
                ++this->successorsIterator;
            }
            return *this;
        }

        template <typename ValueType>
        const typename TotalPayoffGame<ValueType>::iterator TotalPayoffGame<ValueType>::iterator::operator++(int) {
            iterator it = *this;
            ++*this;
            return it;
        }

        template <typename ValueType>
        bool TotalPayoffGame<ValueType>::iterator::operator!=(const TotalPayoffGame<ValueType>::iterator &otherIterator) {
            if (this->forwardSuccessors) {
                return this->successorsIterator != otherIterator.successorsIterator;
            }
            else {
                bool stopped = this->currentRow >= this->rowEnd and otherIterator.currentRow >= otherIterator.rowEnd
                               and (not this->iterate_on_columns) and (not otherIterator.iterate_on_columns);
                bool sameCurrentRow = not this->iterate_on_columns and not otherIterator.iterate_on_columns
                                      and this->currentRow == otherIterator.currentRow
                                      and not (otherIterator.currentRow == 0 and otherIterator.rowEnd == 0); // empty iterator case
                bool sameCurrentColumn = this->iterate_on_columns and otherIterator.iterate_on_columns
                                         and this->currentColumn == otherIterator.currentColumn;
                bool isEqual = stopped or sameCurrentRow or sameCurrentColumn;
                return not isEqual;
            }
        }

        template <typename ValueType>
        uint_fast64_t TotalPayoffGame<ValueType>::iterator::operator*() {
            if (iterate_on_columns) {
                return this->currentColumn;
            }
            else if (forwardSuccessors) {
                return *this->successorsIterator;
            }
            else {
                return this->currentRow;
            }
        }

        template <typename ValueType>
        TotalPayoffGame<ValueType>::forwardSuccessorsP2::forwardSuccessorsP2(
                uint_fast64_t action,
                const storm::storage::SparseMatrix<ValueType> &matrix,
                const sw::game::TotalPayoffGame<ValueType>::ForwardTransitions &forwardTransitions,
                storm::storage::BitVector const &restrictedStateSpace,
                storm::storage::BitVector const &enabledActions)
                : successors(action, matrix, restrictedStateSpace, enabledActions),
                  successorList(forwardTransitions.successors[action]) {}

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator TotalPayoffGame<ValueType>::forwardSuccessorsP2::begin() {
            return iterator(this->successorList.begin(), this->successorList.end(), this->restrictedStateSpace, this->enabledActions);
        }

        template class TotalPayoffGame<double>;
        template class TotalPayoffGame<storm::RationalNumber>;

    }
}
