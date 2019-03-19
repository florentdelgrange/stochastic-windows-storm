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
            ValueType W = -1 * storm::utility::infinity<ValueType>();
            {
                ValueType absoluteWeight;
                for (uint_fast64_t action: this->enabledActions) {
                    absoluteWeight = storm::utility::abs(weights[action]);
                    if (W < absoluteWeight) {
                        W = absoluteWeight;
                    }
                }
            }
            /*
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
             */

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
                    W)
                    .max;
        }

        template<typename ValueType>
        std::vector<ValueType> TotalPayoffGame<ValueType>::minTotalPayoffSup() const {
            std::vector<ValueType> const& weights = rewardModel.getStateActionRewardVector();
            std::vector<ValueType> oppositeWeights(weights.size());
            std::transform(weights.begin(), weights.end(), oppositeWeights.begin(),
                           [](ValueType w) -> ValueType { return w * -1; });
            ValueType W = -1 * storm::utility::infinity<ValueType>();
            {
                ValueType absoluteWeight;
                for (uint_fast64_t action: this->enabledActions) {
                    absoluteWeight = storm::utility::abs(weights[action]);
                    if (W < absoluteWeight) {
                        W = absoluteWeight;
                    }
                }
            }

            std::vector<ValueType> result = acceleratedMaxTotalPayoffInf(
                    storm::Environment(), this->enabledActions, this->restrictedStateSpace, false,
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new forwardSuccessorsP2(this->forwardTransitions.successors[state]) ); },
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new successorsP1(state, this->matrix, this->restrictedStateSpace, this->enabledActions) ); },
                    [](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return storm::utility::zero<ValueType>(); },
                    [&](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return oppositeWeights[s_prime]; },
                    oppositeWeights)
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
            ValueType W = -1 * storm::utility::infinity<ValueType>();
            {
                ValueType absoluteWeight;
                for (uint_fast64_t action: this->enabledActions) {
                    absoluteWeight = storm::utility::abs(weights[action]);
                    if (W < absoluteWeight) {
                        W = absoluteWeight;
                    }
                }
            }

            Values result = maxTotalPayoffInf(
                    storm::Environment(),
                    this->enabledActions, this->restrictedStateSpace,
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new forwardSuccessorsP2(this->forwardTransitions.successors[state]) ); },
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new successorsP1(state, this->matrix, this->restrictedStateSpace, this->enabledActions) ); },
                    [](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return storm::utility::zero<ValueType>(); },
                    [&](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return oppositeWeights[s_prime]; },
                    W);
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
        typename TotalPayoffGame<ValueType>::Values TotalPayoffGame<ValueType>::maxTotalPayoffInf(
                storm::Environment const& env,
                storm::storage::BitVector const& maximizerStateSpace,
                storm::storage::BitVector const& minimizerStateSpace,
                std::function<std::unique_ptr<successors>(uint_fast64_t)> const& maximizerSuccessors,
                std::function<std::unique_ptr<successors>(uint_fast64_t)> const& minimizerSuccessors,
                std::function<ValueType(uint_fast64_t, uint_fast64_t)> const& wMaxToMin,
                std::function<ValueType(uint_fast64_t, uint_fast64_t)> const& wMinToMax,
                ValueType W, bool earlyStopping) const {

            uint_fast64_t numberOfMaxStates = maximizerStateSpace.getNumberOfSetBits();
            uint_fast64_t numberOfMinStates = minimizerStateSpace.getNumberOfSetBits();

            std::vector<uint_fast64_t> oldToNewStateMappingMax = initOldToNewStateMapping(maximizerStateSpace);
            std::vector<uint_fast64_t> oldToNewStateMappingMin = initOldToNewStateMapping(minimizerStateSpace);

            using Values = TotalPayoffGame<ValueType>::Values;
            std::shared_ptr<Values> Y = std::shared_ptr<Values>(new Values()), Y_pre, X, X_pre;
            Y->max = std::vector<ValueType>(numberOfMaxStates, -1 * storm::utility::infinity<ValueType>());
            Y->min = std::vector<ValueType>(numberOfMinStates, -1 * storm::utility::infinity<ValueType>());
            ValueType upperBound = storm::utility::convertNumber<ValueType>(this->restrictedStateSpace.getNumberOfSetBits() - 1) * W;
            ValueType lowerBound = -1 * upperBound;
            // for the vector equality check
            auto precision = storm::utility::convertNumber<ValueType>(env.solver().minMax().getPrecision());
            bool relative = env.solver().minMax().getRelativeTerminationCriterion();

            uint_fast64_t external_counter = 0;
            uint_fast64_t internal_counter = 0;
            clock_t start = clock();

            do {
                ++external_counter;

                // incorporate values of the previous copy of the game into the current copy of the game
                Y_pre = Y;
                Y = std::shared_ptr<Values>(new Values());
                Y->max = std::vector<ValueType>(numberOfMaxStates);
                initNextValues(Y_pre->max, Y->max, maximizerStateSpace, oldToNewStateMappingMax);
                Y->min = std::vector<ValueType>(numberOfMinStates);
                initNextValues(Y_pre->min, Y->min, minimizerStateSpace, oldToNewStateMappingMin);

                X = std::shared_ptr<Values>(new Values());
                X->max = std::vector<ValueType>(numberOfMaxStates, storm::utility::infinity<ValueType>());
                X->min = std::vector<ValueType>(numberOfMinStates, storm::utility::infinity<ValueType>());

                // min-cost reachability
                do {
                    ++ internal_counter;

                    X_pre = X;
                    X = std::shared_ptr<Values>(new Values());
                    // maximizer phase
                    X->max = std::vector<ValueType>(numberOfMaxStates);
                    internalMinCostReachability(X->max, X_pre->min, Y->min,
                                                oldToNewStateMappingMax, oldToNewStateMappingMin,
                                                maximizerStateSpace, maximizerSuccessors, wMaxToMin, true);
                    // minimizer phase
                    X->min = std::vector<ValueType>(numberOfMinStates);
                    internalMinCostReachability(X->min, X_pre->max, Y->max,
                                                oldToNewStateMappingMin, oldToNewStateMappingMax,
                                                minimizerStateSpace, minimizerSuccessors, wMinToMax, false);
                    // lower bound checking phase
                    lowerBoundUpdate(X->max, maximizerStateSpace, oldToNewStateMappingMax, lowerBound);
                    lowerBoundUpdate(X->min, minimizerStateSpace, oldToNewStateMappingMin, lowerBound);

                } while (not valuesEqual(*X, *X_pre, precision, relative));

                Y = X;

                // upper bound checking phase
                upperBoundUpdate(Y->max, maximizerStateSpace, oldToNewStateMappingMax, upperBound);
                upperBoundUpdate(Y->min, minimizerStateSpace, oldToNewStateMappingMin, upperBound);

            } while (not valuesEqual(*Y, *Y_pre, precision, relative) and
                     not (earlyStopping and valuesStrictlyPositive(*Y)));

            clock_t stop = clock();
            double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
            printf("\nTime elapsed: %.5f | internal iterations: %llu | external iterations: %llu \n", elapsed, internal_counter, external_counter);

            Y_pre = Y;
            Y = std::shared_ptr<Values>(new Values());
            Y->max = std::vector<ValueType>(maximizerStateSpace.size());
            backToOldStateMapping(Y_pre->max, Y->max, maximizerStateSpace, oldToNewStateMappingMax);
            Y->min = std::vector<ValueType>(minimizerStateSpace.size());
            backToOldStateMapping(Y_pre->min, Y->min, minimizerStateSpace, oldToNewStateMappingMin);

            return *Y;
        }

        template<typename ValueType>
        typename TotalPayoffGame<ValueType>::Values TotalPayoffGame<ValueType>::acceleratedMaxTotalPayoffInf(
                storm::Environment const &env,
                storm::storage::BitVector const& maximizerStateSpace,
                storm::storage::BitVector const& minimizerStateSpace,
                bool mdpStatesAreMaximizer,
                std::function<std::unique_ptr<successors>(uint_fast64_t)> const &maximizerSuccessors,
                std::function<std::unique_ptr<successors>(uint_fast64_t)> const &minimizerSuccessors,
                std::function<ValueType(uint_fast64_t, uint_fast64_t)> const &wMaxToMin,
                std::function<ValueType(uint_fast64_t, uint_fast64_t)> const &wMinToMax,
                std::vector<ValueType> const& actionsWeight) const {

            auto precision = storm::utility::convertNumber<ValueType>(env.solver().minMax().getPrecision());
            bool relative = env.solver().minMax().getRelativeTerminationCriterion();
            ValueType W = -1 * storm::utility::infinity<ValueType>();
            using Values = TotalPayoffGame<ValueType>::Values;
            uint_fast64_t numberOfMaxStates = maximizerStateSpace.getNumberOfSetBits();
            uint_fast64_t numberOfMinStates = minimizerStateSpace.getNumberOfSetBits();
            uint_fast64_t currentNumberOfStates = 0;
            std::vector<uint_fast64_t> oldToNewStateMappingMax = initOldToNewStateMapping(maximizerStateSpace);
            std::vector<uint_fast64_t> oldToNewStateMappingMin = initOldToNewStateMapping(minimizerStateSpace);

            std::shared_ptr<Values> Y = std::shared_ptr<Values>(new Values()), Y_pre, X, X_pre;
            Y->max = std::vector<ValueType>(numberOfMaxStates, -1 * storm::utility::infinity<ValueType>());
            Y->min = std::vector<ValueType>(numberOfMinStates, -1 * storm::utility::infinity<ValueType>());

            uint_fast64_t external_counter = 0;
            uint_fast64_t internal_counter = 0;
            clock_t start = clock();

            storm::storage::StronglyConnectedComponentDecomposition<ValueType>
            stronglyConnectedComponentDecomposition(this->matrix, this->restrictedStateSpace, this->enabledActions);

            for (const auto &component: stronglyConnectedComponentDecomposition) {
                storm::storage::BitVector sccStates(this->restrictedStateSpace.size(), false);
                storm::storage::BitVector sccActions(this->enabledActions.size(), false);
                for (auto const& state: component) {
                    sccStates.set(state, true);
                    for (uint_fast64_t action = this->enabledActions.getNextSetIndex(this->matrix.getRowGroupIndices()[state]);
                         action < this->matrix.getRowGroupIndices()[state + 1];
                         action = this->enabledActions.getNextSetIndex(action + 1)) {
                        sccActions.set(action, true);
                    }
                }
                currentNumberOfStates += component.size();
                // update the upper and lower bounds according to the greatest absolute weight inside the current SCC
                ValueType absoluteWeight;
                for (uint_fast64_t action: sccActions) {
                    absoluteWeight = storm::utility::abs(actionsWeight[action]);
                    if (W < absoluteWeight) {
                        W = absoluteWeight;
                    }
                }
                ValueType upperBound = storm::utility::convertNumber<ValueType>(currentNumberOfStates - 1) * W;
                ValueType lowerBound = -1 * upperBound;

                storm::storage::BitVector const& sccMaximizerStateSpace = mdpStatesAreMaximizer ? sccStates : sccActions;
                storm::storage::BitVector const& sccMinimizerStateSpace = mdpStatesAreMaximizer ? sccActions : sccStates;
                storm::storage::BitVector frozenMaximizerStateSpace = maximizerStateSpace & ~sccMaximizerStateSpace;
                storm::storage::BitVector frozenMinimizerStateSpace = minimizerStateSpace & ~sccMinimizerStateSpace;

                do {
                    ++ external_counter;
                    // incorporate values of the previous copy of the game into the current copy of the game
                    Y_pre = Y;
                    Y = std::shared_ptr<Values>(new Values()); X = std::shared_ptr<Values>(new Values());
                    Y->max = std::vector<ValueType>(numberOfMaxStates); X->max = std::vector<ValueType>(numberOfMaxStates);
                    initNextValues(Y_pre->max, Y->max, X->max, maximizerStateSpace, sccMaximizerStateSpace, oldToNewStateMappingMax);
                    Y->min = std::vector<ValueType>(numberOfMinStates); X->min = std::vector<ValueType>(numberOfMinStates);
                    initNextValues(Y_pre->min, Y->min, X->min, minimizerStateSpace, sccMinimizerStateSpace, oldToNewStateMappingMin);

                    // min-cost reachability
                    do {
                        ++ internal_counter;

                        X_pre = X;
                        X = std::shared_ptr<Values>(new Values());
                        X->max = std::vector<ValueType>(numberOfMaxStates); X->min = std::vector<ValueType>(numberOfMinStates);
                        initNextMeanCostReachabilityValues(X_pre->max, X->max, frozenMaximizerStateSpace, oldToNewStateMappingMax);
                        initNextMeanCostReachabilityValues(X_pre->min, X->min, frozenMinimizerStateSpace, oldToNewStateMappingMin);
                        // maximizer phase
                        internalMinCostReachability(X->max, X_pre->min, Y->min,
                                                    oldToNewStateMappingMax, oldToNewStateMappingMin,
                                                    sccMaximizerStateSpace, maximizerSuccessors, wMaxToMin, true);
                        // minimizer phase
                        internalMinCostReachability(X->min, X_pre->max, Y->max,
                                                    oldToNewStateMappingMin, oldToNewStateMappingMax,
                                                    sccMinimizerStateSpace, minimizerSuccessors, wMinToMax, false);
                        // lower bound checking phase
                        lowerBoundUpdate(X->max, sccMaximizerStateSpace, oldToNewStateMappingMax, lowerBound);
                        lowerBoundUpdate(X->min, sccMinimizerStateSpace, oldToNewStateMappingMin, lowerBound);

                    } while (not valuesEqual(*X, *X_pre, sccMaximizerStateSpace, sccMinimizerStateSpace,
                                             oldToNewStateMappingMax, oldToNewStateMappingMin, precision, relative));

                    Y = X;

                    // upper bound checking phase
                    upperBoundUpdate(Y->max, sccMaximizerStateSpace, oldToNewStateMappingMax, upperBound);
                    upperBoundUpdate(Y->min, sccMinimizerStateSpace, oldToNewStateMappingMin, upperBound);

                } while (not valuesEqual(*Y, *Y_pre, sccMaximizerStateSpace, sccMinimizerStateSpace,
                                         oldToNewStateMappingMax, oldToNewStateMappingMin, precision, relative));
            }

            clock_t stop = clock();
            double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
            printf("\nTime elapsed: %.5f | internal iterations: %llu | external iterations: %llu \n", elapsed, internal_counter, external_counter);

            Y_pre = Y;
            Y = std::shared_ptr<Values>(new Values());
            Y->max = std::vector<ValueType>(maximizerStateSpace.size());
            backToOldStateMapping(Y_pre->max, Y->max, maximizerStateSpace, oldToNewStateMappingMax);
            Y->min = std::vector<ValueType>(minimizerStateSpace.size());
            backToOldStateMapping(Y_pre->min, Y->min, minimizerStateSpace, oldToNewStateMappingMin);

            return *Y;

        }

        template<typename ValueType>
        void TotalPayoffGame<ValueType>::internalMinCostReachability(
                std::vector<ValueType> &X, std::vector<ValueType> const& X_pre, std::vector<ValueType> const& Y,
                std::vector<uint_fast64_t> const& oldToNewStateMapping,
                std::vector<uint_fast64_t> const& oldToNewStateMappingSucc,
                storm::storage::BitVector const& stateSpace,
                std::function<std::unique_ptr<successors>(uint_fast64_t)> const& successors,
                std::function<ValueType(uint_fast64_t, uint_fast64_t)> const& w,
                bool maximizerPhase) const {

            uint_fast64_t s, s_prime;
            ValueType x;

            for (uint_fast64_t state: stateSpace) {
                s = oldToNewStateMapping[state];
                X[s] = maximizerPhase ? -1 * storm::utility::infinity<ValueType>(): storm::utility::infinity<ValueType>();
                for (uint_fast64_t next_state: *successors(state)) {
                    s_prime = oldToNewStateMappingSucc[next_state];
                    x = w(state, next_state) + storm::utility::min(X_pre[s_prime], Y[s_prime]);
                    if ((maximizerPhase and x > X[s]) or (not maximizerPhase and x < X[s])) {
                        X[s] = x;
                    }
                }
            }
        }

        template<typename ValueType>
        std::vector<uint_fast64_t>
        TotalPayoffGame<ValueType>::initOldToNewStateMapping(storm::storage::BitVector const &stateSpace) const {
            std::vector<uint_fast64_t> oldToNewStateMapping(stateSpace.size());
            uint_fast64_t s = 0;
            for (uint_fast64_t const& state: stateSpace) {
                oldToNewStateMapping[state] = s;
                ++ s;
            }
            return oldToNewStateMapping;
        }

        template<typename ValueType>
        void TotalPayoffGame<ValueType>::initNextValues(const std::vector<ValueType> &Y_pre, std::vector<ValueType> &Y,
                                                        storm::storage::BitVector const &stateSpace,
                                                        std::vector<uint_fast64_t> const &oldToNewStateMapping) const {
            uint_fast64_t s;
            for (uint_fast64_t const& state: stateSpace) {
                s = oldToNewStateMapping[state];
                Y[s] = storm::utility::max<ValueType>(storm::utility::zero<ValueType>(), Y_pre[s]);
            }

        }

        template<typename ValueType>
        void TotalPayoffGame<ValueType>::initNextValues(const std::vector<ValueType> &Y_pre, std::vector<ValueType> &Y,
                                                        std::vector<ValueType> &X,
                                                        storm::storage::BitVector const &stateSpace,
                                                        storm::storage::BitVector const& currentStateSpace,
                                                        std::vector<uint_fast64_t> const &oldToNewStateMapping) const {
            uint_fast64_t s;
            for (uint_fast64_t const& state: stateSpace) {
                s = oldToNewStateMapping[state];
                if (currentStateSpace[state]) {
                    Y[s] = storm::utility::max<ValueType>(storm::utility::zero<ValueType>(), Y_pre[s]);
                    X[s] = storm::utility::infinity<ValueType>();
                }
                else {
                    Y[s] = Y_pre[s];
                    X[s] = Y_pre[s];
                }
            }
        }

        template<typename ValueType>
        void TotalPayoffGame<ValueType>::initNextMeanCostReachabilityValues(const std::vector<ValueType> &X_pre,
                                                                            std::vector<ValueType> &X,
                                                                            storm::storage::BitVector const &frozenStateSpace,
                                                                            std::vector<uint_fast64_t> const &oldToNewStateMapping) const {
            uint_fast64_t s;
            for (uint_fast64_t const& state: frozenStateSpace) {
                s = oldToNewStateMapping[state];
                X[s] = X_pre[s];
            }
        }

        template<typename ValueType>
        void TotalPayoffGame<ValueType>::lowerBoundUpdate(std::vector<ValueType> &X,
                                                          storm::storage::BitVector const &stateSpace,
                                                          std::vector<uint_fast64_t> const &oldToNewStateMapping,
                                                          ValueType const& lowerBound) const {
            uint_fast64_t s;
            for (uint_fast64_t const& state: stateSpace) {
                s = oldToNewStateMapping[state];
                if (X[s] < lowerBound) {
                    X[s] = -1 * storm::utility::infinity<ValueType>();
                }
            }

        }

        template<typename ValueType>
        void TotalPayoffGame<ValueType>::upperBoundUpdate(std::vector<ValueType> &Y,
                                                          storm::storage::BitVector const &stateSpace,
                                                          std::vector<uint_fast64_t> const &oldToNewStateMapping,
                                                          ValueType const& upperBound) const {
            uint_fast64_t s;
            for (uint_fast64_t state: stateSpace) {
                s = oldToNewStateMapping[state];
                if (Y[s] > upperBound) {
                    Y[s] = storm::utility::infinity<ValueType>();
                }
            }
        }

        template<typename ValueType>
        void TotalPayoffGame<ValueType>::backToOldStateMapping(const std::vector<ValueType> &Y_pre,
                                                               std::vector<ValueType> &Y,
                                                               storm::storage::BitVector const &stateSpace,
                                                               std::vector<uint_fast64_t> const &oldToNewStateMapping) const {

            uint_fast64_t s;
            for (uint_fast64_t state = 0; state < stateSpace.size(); ++ state) {
                s = oldToNewStateMapping[state];
                // we set Y_pre[s] to zero as default value if the state does not belong to the restricted state
                // space to avoid conflicts with the strictly positive or negative checking phase
                Y[state] = stateSpace[state] ? Y_pre[s]: storm::utility::zero<ValueType>();
            }
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

        template<typename ValueType>
        bool TotalPayoffGame<ValueType>::vectorEquality(const std::vector<ValueType> &vectorLeft,
                                                        const std::vector<ValueType> &vectorRight,
                                                        storm::storage::BitVector const& stateSpace,
                                                        std::vector<uint_fast64_t> const& oldToNewStateMapping,
                                                        const ValueType &precision, bool relativeError) const {

            STORM_LOG_ASSERT(vectorLeft.size() == vectorRight.size(), "Lengths of vectors does not match.");
            uint_fast64_t s;
            for (uint_fast64_t state: stateSpace) {
                s = oldToNewStateMapping[state];
                if (!this->equalModuloPrecision(vectorLeft[s], vectorRight[s], precision, relativeError)) {
                    return false;
                }
            }
            return true;
        }

        template <typename ValueType>
        bool TotalPayoffGame<ValueType>::valuesEqual(
                typename TotalPayoffGame<ValueType>::Values const& X,
                typename TotalPayoffGame<ValueType>::Values const& Y,
                ValueType precision, bool relativeError) const {

            if (not relativeError) {
                precision *= storm::utility::convertNumber<ValueType>(2.0);
            }

            return this->vectorEquality(X.max, Y.max, precision, relativeError) and
                   this->vectorEquality(X.min, Y.min, precision, relativeError);
        }

        template<typename ValueType>
        bool TotalPayoffGame<ValueType>::valuesEqual(const TotalPayoffGame::Values &X, const TotalPayoffGame::Values &Y,
                                                     storm::storage::BitVector const &maximizerStateSpace,
                                                     storm::storage::BitVector const &minimizerStateSpace,
                                                     std::vector<uint_fast64_t> const& oldToNewStateMappingMax,
                                                     std::vector<uint_fast64_t> const& oldToNewStateMappingMin,
                                                     ValueType precision, bool relativeError) const {
            if (not relativeError) {
                precision *= storm::utility::convertNumber<ValueType>(2.0);
            }

            return this->vectorEquality(X.max, Y.max, maximizerStateSpace, oldToNewStateMappingMax, precision, relativeError) and
                   this->vectorEquality(X.min, Y.min, minimizerStateSpace, oldToNewStateMappingMin, precision, relativeError);
        }

        template <typename ValueType>
        bool TotalPayoffGame<ValueType>::valuesStrictlyPositive(const sw::game::TotalPayoffGame<ValueType>::Values &X) const {
            for (const ValueType &val: X.max) {
                if (val <= 0) {
                    return false;
                }
            }
            for (const ValueType &val: X.min) {
                if (val <= 0) {
                    return false;
                }
            }
            return true;
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
