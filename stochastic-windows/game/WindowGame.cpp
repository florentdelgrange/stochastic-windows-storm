#include <memory>
#include <storm/storage/memorystructure/MemoryStructureBuilder.h>

//
// Created by Florent Delgrange on 2019-01-25.
//

#include "WindowGame.h"
#include "TotalPayoffGame.h"

namespace sw {
    namespace game {

        template<typename ValueType>
        WindowGame<ValueType>::WindowGame(
                storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                std::string const &rewardModelName,
                uint_fast64_t const &l_max,
                storm::storage::BitVector const &restrictedStateSpace,
                storm::storage::BitVector const &enabledActions)
                : MdpGame<ValueType>(mdp, restrictedStateSpace, enabledActions),
                  rewardModelName(rewardModelName),
                  rewardModel(mdp.getRewardModel(rewardModelName)),
                  l_max(l_max) {}

        template<typename ValueType>
        WindowMeanPayoffGame<ValueType>::WindowMeanPayoffGame(
                storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                std::string const &rewardModelName,
                uint_fast64_t const &l_max,
                storm::storage::BitVector const &restrictedStateSpace,
                storm::storage::BitVector const &enabledActions)
                : WindowGame<ValueType>(mdp, rewardModelName, l_max, restrictedStateSpace, enabledActions) {}

        template<typename ValueType>
        WindowMeanPayoffGame<ValueType>::WindowMeanPayoffGame(
                storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                std::string const &rewardModelName,
                storm::storage::BitVector const &restrictedStateSpace,
                storm::storage::BitVector const &enabledActions)
                : WindowGame<ValueType>(mdp, rewardModelName, 0, restrictedStateSpace, enabledActions) {}

        template<typename ValueType>
        WindowParityGame<ValueType>::WindowParityGame(
                const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                std::string const &rewardModelName,
                storm::storage::BitVector const &restrictedStateSpace, storm::storage::BitVector const &enabledActions)
                : WindowGame<ValueType>(mdp, rewardModelName, 0, restrictedStateSpace, enabledActions) {}

        template<typename ValueType>
        storm::storage::BitVector const& WindowGame<ValueType>::getStateSpace() const {
            return this->restrictedStateSpace;
        }

        template<typename ValueType>
        storm::storage::BitVector WindowGame<ValueType>::directFW() const {
            STORM_LOG_ASSERT(l_max > 0, "no maximal window size (>0) set");
            BackwardTransitions backwardTransitions;
            this->initBackwardTransitions(backwardTransitions);
            return directFW(backwardTransitions).winningSet;
        }

        template<typename ValueType>
        WinningSetAndScheduler<ValueType> WindowGame<ValueType>::produceSchedulerForDirectFW() const {
            STORM_LOG_ASSERT(l_max > 0, "no maximal window size (>0) set");
            BackwardTransitions backwardTransitions;
            this->initBackwardTransitions(backwardTransitions);
            return directFW(backwardTransitions, true);
        }

        template<typename ValueType>
        std::unique_ptr<WindowGame<ValueType>>
        WindowGame<ValueType>::restrictToSafePart(storm::storage::BitVector const &safeStates) const {
            BackwardTransitions backwardTransitions;
            this->initBackwardTransitions(backwardTransitions);
            return restrictToSafePart(safeStates, backwardTransitions);
        }


        template<typename ValueType>
        WinningSetAndScheduler<ValueType> WindowGame<ValueType>::directFW(BackwardTransitions &backwardTransitions,
                                                                          bool produceScheduler) const {
            storm::storage::BitVector winGW = this->goodWin().winningSet;
            if (winGW == this->restrictedStateSpace or winGW.empty()) {
                // TODO remove below lines
                bool winningSetEmpty = winGW.empty();
                std::cout << winGW << std::endl;
                if (not produceScheduler or winGW.empty()) {
                    return WinningSetAndScheduler<ValueType>(std::move(winGW));
                } else {
                    return this->goodWin(true);
                }
            } else {
                std::unique_ptr<WindowGame<ValueType>> safeGame = this->restrictToSafePart(winGW, backwardTransitions);
                return safeGame->directFW(backwardTransitions, produceScheduler);
            }
        }

        template<typename ValueType>
        storm::storage::BitVector WindowGame<ValueType>::boundedProblem() const {
            BackwardTransitions backwardTransitions;
            this->initBackwardTransitions(backwardTransitions);

            storm::storage::BitVector W_bp(this->restrictedStateSpace.size(), false);
            storm::storage::BitVector L = unbOpenWindow().p1States;
            storm::storage::BitVector remainingSet = this->restrictedStateSpace;
            while (remainingSet != L) {
                W_bp = this->attractorsP1(this->restrictedStateSpace & ~L, backwardTransitions);
                remainingSet = this->restrictedStateSpace & ~W_bp;
                std::unique_ptr<WindowGame<ValueType>> badSubGame = this->restrict(remainingSet);
                L = badSubGame->unbOpenWindow().p1States;
            }
            return W_bp;
        }

        template<typename ValueType>
        storm::storage::BitVector WindowGame<ValueType>::directBoundedProblem(boost::optional<storm::storage::Scheduler<ValueType>&> const& scheduler) const {
            GameStates loosingRegion = this->unbOpenWindow();
            storm::storage::BitVector safeStates = this->restrictedStateSpace & ~loosingRegion.p1States;
            // if a scheduler is provided, fill in it according to the safe region computed
            if (scheduler) {
                storm::storage::BitVector safeActions = this->enabledActions & ~loosingRegion.p2States;
                for (uint_fast64_t const& state: safeStates) {
                    // set an arbitrary safe action as being the one chosen by the scheduler
                    uint_fast64_t action = safeActions.getNextSetIndex(this->matrix.getRowGroupIndices()[state]);
                    for (uint_fast64_t memoryState = 0; memoryState < scheduler->getNumberOfMemoryStates(); ++ memoryState) {
                        scheduler->setChoice(action - this->matrix.getRowGroupIndices()[state], state, memoryState);
                    }
                }
            }
            return safeStates;
        }

        template<typename ValueType>
        WinningSetAndScheduler<ValueType> WindowGame<ValueType>::goodWin(bool produceScheduler) const {
            // default goodWin: empty set
            return WinningSetAndScheduler<ValueType>(std::move(storm::storage::BitVector(this->restrictedStateSpace.size(), false)));
        }

        template<typename ValueType>
        WinningSetAndScheduler<ValueType> WindowMeanPayoffGame<ValueType>::goodWin(bool produceScheduler) const {
            uint_fast64_t numberOfStates = this->restrictedStateSpace.getNumberOfSetBits();
            // C[l][s] is the best sum that can be ensured from state s in at most l steps
            std::vector<std::vector<ValueType>> C(this->l_max, std::vector<ValueType>(numberOfStates));
            // To avoid C to have a size of l_max X number of states (but rather l_max X number of restricted states)
            std::vector<uint_fast64_t> oldToNewStateMapping(this->mdp.getNumberOfStates());
            std::vector<uint_fast64_t> const& stateActionIndices = this->matrix.getRowGroupIndices();
            std::vector<ValueType> const& weight = this->rewardModel.getStateActionRewardVector();
            // winning set
            storm::storage::BitVector winningSet(this->mdp.getNumberOfStates(), false);

            std::unique_ptr<std::vector<std::vector<uint_fast64_t>>> bestActions;
            if (produceScheduler) {
                bestActions = std::make_unique<std::vector<std::vector<uint_fast64_t>>>(this->l_max, std::vector<uint_fast64_t>(numberOfStates));
            }

            // Initialization of the matrix C
            uint_fast64_t s = 0;
            for (uint_fast64_t state: this->restrictedStateSpace) {
                uint_fast64_t action = this->enabledActions.getNextSetIndex(stateActionIndices[state]);
                C[0][s] = -1 * storm::utility::infinity<ValueType>();
                // iterate on enabled actions of state s
                for (action = this->enabledActions.getNextSetIndex(action);
                     action < stateActionIndices[state + 1];
                     action = this->enabledActions.getNextSetIndex(action + 1)) {
                    if (weight[action] > C[0][s]) {
                        C[0][s] = weight[action];
                        if (produceScheduler) {
                            (*bestActions)[0][s] = action;
                        }
                    }
                }
                oldToNewStateMapping[state] = s;
                ++ s;
            }
            // given a number of steps i and an action a, retrieves the worst successor value of a in C[i - 1]
            auto worstSuccessorValue = [&] (uint_fast64_t const& i, uint_fast64_t const& action) -> ValueType {
                auto worstValue = storm::utility::infinity<ValueType>();
                // We assume that all entries (successors) of the row corresponding to the current action are in
                // the restricted state space
                for (auto const &successorEntry : this->matrix.getRow(action)) {
                    uint_fast64_t s_prime = oldToNewStateMapping[successorEntry.getColumn()];
                    if (C[i - 1][s_prime] < worstValue) {
                        worstValue = C[i - 1][s_prime];
                    }
                }
                return worstValue;
            };

            // Fill in C up to l_max
            for (uint_fast64_t i = 1; i < this->l_max; ++ i) {
                for (uint_fast64_t state: this->restrictedStateSpace) {
                    s = oldToNewStateMapping[state];
                    C[i][s] = -1 * storm::utility::infinity<ValueType>();
                    for (uint_fast64_t action = this->enabledActions.getNextSetIndex(stateActionIndices[state]);
                         action < stateActionIndices[state + 1];
                         action = this->enabledActions.getNextSetIndex(action + 1)) {
                        auto actionValue = storm::utility::max<ValueType>(weight[action], weight[action] + worstSuccessorValue(i, action));
                        if (actionValue > C[i][s]) {
                            C[i][s] = actionValue;
                            if (produceScheduler) {
                                (*bestActions)[i][s] = action;
                            }
                        }
                    }
                    // Construct the winning set
                    if (i == this->l_max - 1 && C[i][s] >= 0) {
                        winningSet.set(state, true);
                    }
                }
            }
            std::unique_ptr<storm::storage::Scheduler<ValueType>> scheduler;
            if (produceScheduler) {
                storm::storage::MemoryStructureBuilder<ValueType> memoryBuilder(this->l_max, this->mdp);
                storm::storage::BitVector unsafeStates = ~this->restrictedStateSpace;
                storm::Environment env;
                auto precision = storm::utility::convertNumber<ValueType>(env.solver().minMax().getPrecision());
                bool relative = env.solver().minMax().getRelativeTerminationCriterion();

                for (uint_fast64_t l = 0; l < this->l_max - 1; ++ l) {
                    // states that does not belong to the winning region are omitted
                    memoryBuilder.setTransition(l, 0, unsafeStates);
                    storm::storage::BitVector continueActions(this->mdp.getNumberOfChoices(), false);
                    for (uint_fast64_t const& state: this->restrictedStateSpace) {
                        s = oldToNewStateMapping[state];
                        for (uint_fast64_t action = this->enabledActions.getNextSetIndex(stateActionIndices[state]);
                             action < stateActionIndices[state + 1];
                             action = this->enabledActions.getNextSetIndex(action + 1)) {
                            auto actionValue = storm::utility::max<ValueType>(
                                    weight[action],
                                    weight[action] + worstSuccessorValue(this->l_max - 1 - l, action));
                            if (not storm::utility::vector::equalModuloPrecision<ValueType>(C[0][s], actionValue, precision, relative)) {
                                continueActions.set(action, true);
                            }
                        }
                    }
                    // choosing an action that does not ensure to make the current sum positive
                    memoryBuilder.setTransition(l, l + 1, this->restrictedStateSpace, continueActions);
                    // good reset: the sum is maximal by playing the action in less than l_max steps
                    // OR the arrival state and/or the action played does not belong to this game
                    memoryBuilder.setTransition(l, 0, storm::storage::BitVector(this->mdp.getNumberOfStates(), true), ~continueActions);
                }
                // good reset: the current sum becomes positive within l_max steps;
                // bad  reset: the current sum remains negative within l_max steps
                memoryBuilder.setTransition(this->l_max - 1, 0, storm::storage::BitVector(this->mdp.getNumberOfStates(), true));
                storm::storage::MemoryStructure const& memory = memoryBuilder.build();
                scheduler = std::unique_ptr<storm::storage::Scheduler<ValueType>>(
                        new storm::storage::Scheduler<ValueType>(this->mdp.getNumberOfStates(), memory)
                );
                for (uint_fast64_t l = 0; l < this->l_max; ++ l) {
                    for (uint_fast64_t const& state: this->restrictedStateSpace) {
                        s = oldToNewStateMapping[state];
                        scheduler->setChoice((*bestActions)[this->l_max - 1 - l][s] - this->matrix.getRowGroupIndices()[state], state, l);
                    }
                }
            }
            return produceScheduler ? WinningSetAndScheduler<ValueType>(std::move(winningSet), std::move(scheduler))
                                    : WinningSetAndScheduler<ValueType>(std::move(winningSet));
        }

        template<typename ValueType>
        GameStates WindowMeanPayoffGame<ValueType>::unbOpenWindow() const {

            std::shared_ptr<GameStates> L = std::shared_ptr<GameStates>(new GameStates()), L_pre;
            L->p1States = storm::storage::BitVector(this->restrictedStateSpace.size(), false);
            L->p2States = storm::storage::BitVector(this->enabledActions.size(), false);
            do {
                L_pre = L;
                L = std::shared_ptr<GameStates>(new GameStates());
                storm::storage::BitVector remainingStateSpace = this->restrictedStateSpace & ~L_pre->p1States;
                storm::storage::BitVector remainingEnabledActions = this->enabledActions & ~L_pre->p2States;
                TotalPayoffGame<ValueType> totalPayoffGame(this->mdp,
                                                           this->rewardModelName,
                                                           remainingStateSpace,
                                                           remainingEnabledActions);

                BackwardTransitions backwardTransitions;
                totalPayoffGame.initBackwardTransitions(backwardTransitions);

                GameStates negSupTp = totalPayoffGame.negSupTP();
                GameStates attractorsNegSupTp = totalPayoffGame.attractorsP2(negSupTp, backwardTransitions);
                L->p1States = L_pre->p1States | attractorsNegSupTp.p1States;
                L->p2States = L_pre->p2States | attractorsNegSupTp.p2States;
            } while (L->p1States != L_pre->p1States or L->p2States != L_pre->p2States);

            return *L;
        }

        template<typename ValueType>
        GameStates WindowParityGame<ValueType>::unbOpenWindow() const {

            std::shared_ptr<GameStates> L = std::shared_ptr<GameStates>(new GameStates()), L_pre;
            L->p1States = storm::storage::BitVector(this->restrictedStateSpace.size(), false);
            L->p2States = storm::storage::BitVector(this->enabledActions.size(), false);
            do {
                L_pre = L;
                L = std::shared_ptr<GameStates>(new GameStates());
                storm::storage::BitVector remainingStateSpace = this->restrictedStateSpace & ~L_pre->p1States;
                storm::storage::BitVector remainingEnabledActions = this->enabledActions & ~L_pre->p2States;
                WeakParityGame<ValueType> weakParityGame(this->mdp,
                                                         this->rewardModelName,
                                                         remainingStateSpace,
                                                         remainingEnabledActions);
                BackwardTransitions backwardTransitions;
                weakParityGame.initBackwardTransitions(backwardTransitions);

                GameStates weakcoParity = weakParityGame.weakParity().winningSetP2;
                GameStates attractorsWeakcoParity = weakParityGame.attractorsP2(weakcoParity, backwardTransitions);
                L->p1States = L_pre->p1States | attractorsWeakcoParity.p1States;
                L->p2States = L_pre->p2States | attractorsWeakcoParity.p2States;
            } while (L->p1States != L_pre->p1States or L->p2States != L_pre->p2States);

            return *L;
        }

        template<typename ValueType>
        std::unique_ptr<WindowGame<ValueType>>
        WindowMeanPayoffGame<ValueType>::restrictToSafePart(
                storm::storage::BitVector const& safeStates,
                BackwardTransitions& backwardTransitions) const {

            storm::storage::BitVector badStates = (~safeStates) & this->restrictedStateSpace;
            storm::storage::BitVector restrictedStateSpace = this->restrictedStateSpace & safeStates;
            storm::storage::BitVector enabledActions = this->enabledActions;
            // disable all enabled actions of bad states
            for (uint_fast64_t state: badStates) {
                backwardTransitions.numberOfEnabledActions[state] = 0;
                for (uint_fast64_t action = this->enabledActions.getNextSetIndex(this->matrix.getRowGroupIndices()[state]);
                     action < this->matrix.getRowGroupIndices()[state + 1];
                     action = this->enabledActions.getNextSetIndex(action + 1)) {
                    enabledActions.set(action, false);
                }
            }

            // Initialize a stack to iterate on bad states and its (P2) attractors
            std::forward_list<uint_fast64_t> stack(badStates.begin(), badStates.end());
            uint_fast64_t currentBadState, state;
            while (!stack.empty()) {
                currentBadState = stack.front();
                stack.pop_front();
                for (const auto &action : backwardTransitions.statePredecessors[currentBadState]) {
                    if (enabledActions[action]) {
                        enabledActions.set(action, false);
                        state = backwardTransitions.actionPredecessor[action];
                        backwardTransitions.numberOfEnabledActions[state] -= 1;
                        if (! backwardTransitions.numberOfEnabledActions[state]) {
                            restrictedStateSpace.set(state, false);
                            stack.push_front(state);
                        }
                    }
                }
            }

            return std::unique_ptr<WindowGame<ValueType>>(
                    new WindowMeanPayoffGame<ValueType>(this->mdp,
                                                        this->rewardModelName,
                                                        this->l_max,
                                                        std::move(restrictedStateSpace),
                                                        std::move(enabledActions))
            );
        }

        template<typename ValueType>
        std::unique_ptr<WindowGame<ValueType>>
        WindowParityGame<ValueType>::restrictToSafePart(
                storm::storage::BitVector const& safeStates,
                BackwardTransitions& backwardTransitions) const {

            storm::storage::BitVector badStates = (~safeStates) & this->restrictedStateSpace;
            storm::storage::BitVector restrictedStateSpace = this->restrictedStateSpace & safeStates;
            storm::storage::BitVector enabledActions = this->enabledActions;
            // disable all enabled actions of bad states
            for (uint_fast64_t state: badStates) {
                backwardTransitions.numberOfEnabledActions[state] = 0;
                for (uint_fast64_t action = this->enabledActions.getNextSetIndex(this->matrix.getRowGroupIndices()[state]);
                     action < this->matrix.getRowGroupIndices()[state + 1];
                     action = this->enabledActions.getNextSetIndex(action + 1)) {
                    enabledActions.set(action, false);
                }
            }

            // Initialize a stack to iterate on bad states and its (P2) attractors
            std::forward_list<uint_fast64_t> stack(badStates.begin(), badStates.end());
            uint_fast64_t currentBadState, state;
            while (!stack.empty()) {
                currentBadState = stack.front();
                stack.pop_front();
                for (const auto &action : backwardTransitions.statePredecessors[currentBadState]) {
                    if (enabledActions[action]) {
                        enabledActions.set(action, false);
                        state = backwardTransitions.actionPredecessor[action];
                        backwardTransitions.numberOfEnabledActions[state] -= 1;
                        if (! backwardTransitions.numberOfEnabledActions[state]) {
                            restrictedStateSpace.set(state, false);
                            stack.push_front(state);
                        }
                    }
                }
            }

            return std::unique_ptr<WindowGame<ValueType>>(
                    new WindowParityGame<ValueType>(this->mdp,
                                                    this->rewardModelName,
                                                    // this->l_max,
                                                    std::move(restrictedStateSpace),
                                                    std::move(enabledActions))
            );
        }

        template <typename ValueType>
        std::unique_ptr<WindowGame<ValueType>>
        WindowMeanPayoffGame<ValueType>::restrict(storm::storage::BitVector const &restrictedStateSpace) const {
            storm::storage::BitVector removedStates = this->restrictedStateSpace & ~restrictedStateSpace;
            storm::storage::BitVector enabledActions = this->enabledActions;
            for (uint_fast64_t state: removedStates) {
                for (uint_fast64_t action = this->enabledActions.getNextSetIndex(this->matrix.getRowGroupIndices()[state]);
                     action < this->matrix.getRowGroupIndices()[state + 1];
                     action = this->enabledActions.getNextSetIndex(action + 1)) {
                    enabledActions.set(action, false);
                }
            }
            return std::unique_ptr<WindowGame<ValueType>>(
                    new WindowMeanPayoffGame<ValueType>(this->mdp,
                                                        this->rewardModelName,
                                                        this->l_max,
                                                        restrictedStateSpace,
                                                        std::move(enabledActions))
            );
        }

        template <typename ValueType>
        std::unique_ptr<WindowGame<ValueType>>
        WindowParityGame<ValueType>::restrict(storm::storage::BitVector const &restrictedStateSpace) const {
            storm::storage::BitVector removedStates = this->restrictedStateSpace & ~restrictedStateSpace;
            storm::storage::BitVector enabledActions = this->enabledActions;
            for (uint_fast64_t state: removedStates) {
                for (uint_fast64_t action = this->enabledActions.getNextSetIndex(this->matrix.getRowGroupIndices()[state]);
                     action < this->matrix.getRowGroupIndices()[state + 1];
                     action = this->enabledActions.getNextSetIndex(action + 1)) {
                    enabledActions.set(action, false);
                }
            }
            return std::unique_ptr<WindowGame<ValueType>>(
                    new WindowParityGame<ValueType>(this->mdp,
                                                    this->rewardModelName,
                                                    // this->l_max,
                                                    restrictedStateSpace,
                                                    std::move(enabledActions))
            );
        }

        template class WindowGame<double>;
        template class WindowGame<storm::RationalNumber>;
        template class WindowMeanPayoffGame<double>;
        template class WindowMeanPayoffGame<storm::RationalNumber>;
        template class WindowParityGame<double>;

    }
}