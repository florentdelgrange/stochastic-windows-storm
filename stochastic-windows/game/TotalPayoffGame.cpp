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
                const storm::storage::BitVector &enabledActions)
                : MdpGame<ValueType>(mdp, restrictedStateSpace, enabledActions),
                  rewardModelName(rewardModelName),
                  rewardModel(mdp.getRewardModel(rewardModelName)) {

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

        template<typename ValueType>
        std::vector<ValueType> TotalPayoffGame<ValueType>::maxTotalPayoffInf() const {
            std::vector<ValueType> const& weights = rewardModel.getStateActionRewardVector();
            std::vector<ValueType> absoluteWeights(weights.size());
            std::transform(weights.begin(), weights.end(), absoluteWeights.begin(),
                           [](ValueType w) -> ValueType { return storm::utility::abs(w); });
            {
                std::function<std::unique_ptr<successors>(uint_fast64_t)> p2TransitionFunction =
                        [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new forwardSuccessorsP2(this->forwardTransitions.successors[state])); };
                std::function<std::unique_ptr<successors>(uint_fast64_t)> p1TransitionFunction =
                        [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                            return std::unique_ptr<successors>(
                                    new successorsP1(state, this->matrix, this->restrictedStateSpace,
                                                     this->enabledActions));
                        };


                for (uint_fast64_t s: this->restrictedStateSpace) {
                    successors &succ = *p1TransitionFunction(s);
                    auto it = succ.begin();
                    auto end = succ.end();
                    std::cout << "succ of state " << s << "=[";
                    for (uint_fast64_t a: *p1TransitionFunction(s)) {
                        std::cout << a << ", ";
                    }
                    std::cout << "]" << std::endl;
                }

                for (uint_fast64_t a: this->enabledActions) {
                    successors &succ = *p2TransitionFunction(a);
                    auto it = succ.begin();
                    auto end = succ.end();
                    std::cout << "succ of action " << a << "=[";
                    for (uint_fast64_t s: *p2TransitionFunction(a)) {
                        std::cout << s << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
            }

            return maxTotalPayoffInf(
                    storm::Environment(),
                    this->restrictedStateSpace,
                    this->enabledActions,
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new successorsP1(state, this->matrix, this->restrictedStateSpace, this->enabledActions) ); },
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new forwardSuccessorsP2(this->forwardTransitions.successors[state]) ); },
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
                    storm::Environment(), this->enabledActions, this->restrictedStateSpace,
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new forwardSuccessorsP2(this->forwardTransitions.successors[state]) ); },
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

            Values result = maxTotalPayoffInf(
                    storm::Environment(),
                    this->enabledActions, this->restrictedStateSpace,
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new forwardSuccessorsP2(this->forwardTransitions.successors[state]) ); },
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

        /**
         * -------------------------------------------------------------------------------------------------------------
         * Iterators implementation
         * -------------------------------------------------------------------------------------------------------------
         */

        template <typename ValueType>
        TotalPayoffGame<ValueType>::iterator::~iterator() = default;

        template<typename ValueType>
        TotalPayoffGame<ValueType>::successorsIterator::successorsIterator(std::unique_ptr<iterator> concreteIterator)
        : concreteIterator(std::move(concreteIterator)) {}

        template<typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator &TotalPayoffGame<ValueType>::successorsIterator::operator++() {
            return ++*this->concreteIterator;
        }

        template<typename ValueType>
        bool TotalPayoffGame<ValueType>::successorsIterator::operator!=(const iterator &otherIterator) {
            return *this->concreteIterator != otherIterator;
        }

        template<typename ValueType>
        uint_fast64_t TotalPayoffGame<ValueType>::successorsIterator::operator*() {
            return **this->concreteIterator;
        }

        template<typename ValueType>
        uint_fast64_t TotalPayoffGame<ValueType>::successorsIterator::operator*() const {
            return **this->concreteIterator;
        }

        template<typename ValueType>
        bool TotalPayoffGame<ValueType>::successorsIterator::end() const {
            return this->concreteIterator->end();
        }

        template <typename ValueType>
        TotalPayoffGame<ValueType>::successors::~successors() = default;

        template <typename ValueType>
        TotalPayoffGame<ValueType>::iteratorP1::iteratorP1(uint_fast64_t rowBegin, uint_fast64_t rowEnd,
                                                           storm::storage::BitVector const &enabledActions)
                                                           : currentRow(enabledActions.getNextSetIndex(rowBegin)),
                                                             rowEnd(rowEnd),
                                                             enabledActions(enabledActions) {}

        template<typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator &TotalPayoffGame<ValueType>::iteratorP1::operator++() {
            if (this->currentRow < this->rowEnd) {
                this->currentRow = this->enabledActions.getNextSetIndex(currentRow + 1);
            }
            return *this;
        }

        template<typename ValueType>
        bool TotalPayoffGame<ValueType>::iteratorP1::operator!=(const iterator &otherIterator) {
            /*
                bool end = this->end() and otherIterator.end();
                bool sameCurrentRow = this->currentRow == *otherIterator and not otherIterator.end();
                return not (end or sameCurrentRow);
            */
            return not (this->end() and otherIterator.end());
        }

        template<typename ValueType>
        bool TotalPayoffGame<ValueType>::iteratorP1::end() const {
            return this->currentRow >= this->rowEnd;
        }

        template <typename ValueType>
        uint_fast64_t TotalPayoffGame<ValueType>::iteratorP1::operator*() {
            return this->currentRow;
        }

        template <typename ValueType>
        uint_fast64_t TotalPayoffGame<ValueType>::iteratorP1::operator*() const {
            return this->currentRow;
        }

        template <typename ValueType>
        TotalPayoffGame<ValueType>::successorsP1::successorsP1(
                uint_fast64_t state,
                const storm::storage::SparseMatrix<ValueType> &matrix,
                storm::storage::BitVector const& restrictedStateSpace,
                storm::storage::BitVector const& enabledActions)
                : state(state), matrix(matrix), restrictedStateSpace(restrictedStateSpace), enabledActions(enabledActions){}

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::successorsIterator TotalPayoffGame<ValueType>::successorsP1::begin() {
            return restrictedStateSpace[this->state] ?
                successorsIterator(
                        std::unique_ptr<iterator>(
                                new iteratorP1(this->matrix.getRowGroupIndices()[this->state], this->matrix.getRowGroupIndices()[this->state + 1], this->enabledActions)
                        )
                )
                : successorsIterator(std::unique_ptr<iterator>(new iteratorP1(0, 0, this->enabledActions)));
        }

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::successorsIterator TotalPayoffGame<ValueType>::successorsP1::end() {
            return successorsIterator(std::unique_ptr<iterator>(new iteratorP1(0, 0, this->enabledActions)));
        }

        template <typename ValueType>
        TotalPayoffGame<ValueType>::forwardIteratorP2::forwardIteratorP2(
                typename std::forward_list<uint_fast64_t>::const_iterator successorsIteratorBegin,
                typename std::forward_list<uint_fast64_t>::const_iterator successorsIteratorEnd)
                : successorsIterator(successorsIteratorBegin), ptr_end(successorsIteratorEnd) {}

        template<typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator &TotalPayoffGame<ValueType>::forwardIteratorP2::operator++() {
            if (this->successorsIterator != this->ptr_end) {
                ++this->successorsIterator;
            }
            return *this;
        }

        template<typename ValueType>
        bool TotalPayoffGame<ValueType>::forwardIteratorP2::operator!=(const iterator &otherIterator) {
            return not (this->end() and otherIterator.end());
        }

        template <typename ValueType>
        uint_fast64_t TotalPayoffGame<ValueType>::forwardIteratorP2::operator*() {
            return *this->successorsIterator;
        }

        template<typename ValueType>
        uint_fast64_t TotalPayoffGame<ValueType>::forwardIteratorP2::operator*() const {
            return *this->successorsIterator;
        }

        template<typename ValueType>
        bool TotalPayoffGame<ValueType>::forwardIteratorP2::end() const {
            return this->successorsIterator == this->ptr_end;
        }

        template <typename ValueType>
        TotalPayoffGame<ValueType>::forwardSuccessorsP2::forwardSuccessorsP2(
                std::forward_list<uint_fast64_t> const &successorList): successorList(successorList) {}

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::successorsIterator TotalPayoffGame<ValueType>::forwardSuccessorsP2::begin() {
            return successorsIterator(
                    std::unique_ptr<iterator>(new forwardIteratorP2(this->successorList.begin(), this->successorList.end()))
            );
        }

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::successorsIterator TotalPayoffGame<ValueType>::forwardSuccessorsP2::end() {
            return successorsIterator(
                    std::unique_ptr<iterator>(new forwardIteratorP2(this->successorList.end(), this->successorList.end()))
            );
        }

        template class TotalPayoffGame<double>;
        template class TotalPayoffGame<storm::RationalNumber>;

    }
}
