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
        storm::storage::BitVector const& WindowGame<ValueType>::getStateSpace() const {
            return this->restrictedStateSpace;
        }

        template<typename ValueType>
        storm::storage::BitVector WindowGame<ValueType>::directFW() const {
            BackwardTransitions backwardTransitions;
            this->initBackwardTransitions(backwardTransitions);
            return directFW(backwardTransitions);
        }

        template<typename ValueType>
        std::unique_ptr<WindowGame<ValueType>>
        WindowGame<ValueType>::restrictToSafePart(storm::storage::BitVector const &safeStates) const {
            BackwardTransitions backwardTransitions;
            this->initBackwardTransitions(backwardTransitions);
            return restrictToSafePart(safeStates, backwardTransitions);
        }


        template<typename ValueType>
        storm::storage::BitVector WindowGame<ValueType>::directFW(BackwardTransitions &backwardTransitions) const {
            storm::storage::BitVector winGW = this->goodWin();
            if (winGW == this->restrictedStateSpace or winGW.empty()) {
                return winGW;
            }
            else {
                std::unique_ptr<WindowGame<ValueType>> safeGame = this->restrictToSafePart(winGW, backwardTransitions);
                return safeGame->directFW(backwardTransitions);
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
                std::unique_ptr<WindowGame> badSubGame = this->unsafeRestrict(remainingSet);
                L = badSubGame->unbOpenWindow().p1States;
            }
            return W_bp;
        }

        template<typename ValueType>
        storm::storage::BitVector WindowMeanPayoffGame<ValueType>::goodWin() const {
            // C[l][s] is the best sum that can be ensured from state s in at most l steps
            std::vector<std::vector<ValueType>> C(this->l_max, std::vector<ValueType>(this->restrictedStateSpace.getNumberOfSetBits()));
            // To avoid C to have a size of l_max X number of states (but rather l_max X number of restricted states)
            std::vector<uint_fast64_t> oldToNewStateMapping(this->mdp.getNumberOfStates());
            std::vector<uint_fast64_t> const& stateActionIndices = this->matrix.getRowGroupIndices();
            std::vector<ValueType> const& weight = this->rewardModel.getStateActionRewardVector();
            // to deduct the number of enabled actions of each state
            std::vector<uint_fast64_t> const& numberOfEnabledActionsBeforeIndices = this->enabledActions.getNumberOfSetBitsBeforeIndices();
            // winning set
            storm::storage::BitVector winningSet(this->mdp.getNumberOfStates(), false);

            // Initialization of the matrix C
            uint_fast64_t s = 0;
            for (uint_fast64_t state: this->restrictedStateSpace) {
                uint_fast64_t action = this->enabledActions.getNextSetIndex(stateActionIndices[state]);
                C[0][s] = weight[action];
                // iterate on enabled actions of state s
                for (action = this->enabledActions.getNextSetIndex(action + 1);
                     action < stateActionIndices[state + 1];
                     action = this->enabledActions.getNextSetIndex(action + 1)) {
                    if (weight[action] > C[0][s]) {
                        C[0][s] = weight[action];
                    }
                }
                oldToNewStateMapping[state] = s;
                ++ s;
            }
            // Fill in C up to l_max
            for (uint_fast64_t i = 1; i < this->l_max; ++ i) {
                for (uint_fast64_t state: this->restrictedStateSpace) {
                    s = oldToNewStateMapping[state];
                    std::vector<ValueType> actionValues;
                    uint_fast64_t numberOfEnabledActions;
                    // if the considered state is the last of the enumeration of states or if there is no more enabled
                    // action for states with greater indices than the current one.
                    if (state == this->mdp.getNumberOfStates() - 1 or stateActionIndices[state + 1] >= numberOfEnabledActionsBeforeIndices.size()) {
                        numberOfEnabledActions = this->enabledActions.getNumberOfSetBits() -
                                                 numberOfEnabledActionsBeforeIndices[stateActionIndices[state]];
                    }
                    else {
                        numberOfEnabledActions = numberOfEnabledActionsBeforeIndices[stateActionIndices[state + 1]] -
                                                 numberOfEnabledActionsBeforeIndices[stateActionIndices[state]];
                    }
                    actionValues.reserve(numberOfEnabledActions);
                    for (uint_fast64_t action = this->enabledActions.getNextSetIndex(stateActionIndices[state]);
                         action < stateActionIndices[state + 1];
                         action = this->enabledActions.getNextSetIndex(action + 1)) {
                        std::vector<ValueType> successorValues;
                        // We assume that all entries (successors) of the row corresponding to the current action are in
                        // the restricted state space
                        successorValues.reserve(this->matrix.getRow(action).getNumberOfEntries());
                        for (auto const &successorEntry : this->matrix.getRow(action)) {
                            uint_fast64_t s_prime = oldToNewStateMapping[successorEntry.getColumn()];
                            successorValues.push_back(C[i - 1][s_prime]);
                        }
                        ValueType worstSuccessorValue = storm::utility::minimum(successorValues);
                        actionValues.push_back(storm::utility::max<ValueType>(weight[action], weight[action] + worstSuccessorValue));
                    }
                    C[i][s] = storm::utility::maximum(actionValues);

                    // Construct the winning set
                    if (i == this->l_max - 1 && C[i][s] >= 0) {
                        winningSet.set(state, true);
                    }
                }
            }
            return winningSet;
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

        template <typename ValueType>
        std::unique_ptr<WindowGame<ValueType>>
        WindowMeanPayoffGame<ValueType>::unsafeRestrict(storm::storage::BitVector const &restrictedStateSpace) const {
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
                                                        enabledActions)
            );
        }

        template class WindowGame<double>;
        template class WindowGame<storm::RationalNumber>;
        template class WindowMeanPayoffGame<double>;
        template class WindowMeanPayoffGame<storm::RationalNumber>;

    }
}