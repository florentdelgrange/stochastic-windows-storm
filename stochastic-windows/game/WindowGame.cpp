//
// Created by Florent Delgrange on 2019-01-25.
//

#include "WindowGame.h"

namespace sw {
    namespace Game {

        template<typename ValueType>
        WindowGame<ValueType>::WindowGame(
                storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                std::string const &rewardModelName,
                uint_fast64_t const &l_max,
                storm::storage::BitVector const &restrictedStateSpace,
                storm::storage::BitVector const &enabledActions)
                : mdp(mdp),
                  matrix(mdp.getTransitionMatrix()),
                  rewardModelName(rewardModelName),
                  rewardModel(mdp.getRewardModel(rewardModelName)),
                  l_max(l_max),
                  restrictedStateSpace(restrictedStateSpace),
                  enabledActions(enabledActions) {
                      assert(restrictedStateSpace.size() == mdp.getNumberOfStates());
                      assert(enabledActions.size() == this->matrix.getRowCount());
                  }

        template<typename ValueType>
        WindowMeanPayoffGame<ValueType>::WindowMeanPayoffGame(
                storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                std::string const &rewardModelName,
                uint_fast64_t const &l_max,
                storm::storage::BitVector const &restrictedStateSpace,
                storm::storage::BitVector const &enabledActions)
                : WindowGame<ValueType>(mdp, rewardModelName, l_max, restrictedStateSpace, enabledActions) {}

        template<typename ValueType>
        WindowParityGame<ValueType>::WindowParityGame(
                storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                std::string const &rewardModelName,
                uint_fast64_t const &l_max,
                storm::storage::BitVector const &restrictedStateSpace,
                storm::storage::BitVector const &enabledActions)
                : WindowGame<ValueType>(mdp, rewardModelName, l_max, restrictedStateSpace, enabledActions) {}

        template<typename ValueType>
        std::vector<storm::storage::BitVector> WindowGame<ValueType>::getSuccessorStates() {

            std::vector<storm::storage::BitVector> successors(this->mdp.getNumberOfChoices(),
                    storm::storage::BitVector(this->mdp.getNumberOfStates(), false));
            for (uint_fast64_t action: this->enabledActions) {
                for (auto const& successorEntry : this->matrix.getRow(action)) {
                    if (this->restrictedStateSpace[successorEntry.getColumn()]) {
                        successors[action].set(successorEntry.getColumn(), true);
                    }
                }
            }
            return successors;
        }

        template<typename ValueType>
        storm::storage::BitVector WindowGame<ValueType>::directFWMP() {
            storage::PredecessorsSquaredLinkedList<ValueType> predList(this->matrix, this->restrictedStateSpace, this->enabledActions);
            return directFWMP(predList);
        }

        template<typename ValueType>
        std::unique_ptr<WindowGame<ValueType>>
        WindowGame<ValueType>::restrictToSafePart(storm::storage::BitVector const &safeStates) {
            storage::PredecessorsSquaredLinkedList<ValueType> predList(this->matrix, this->restrictedStateSpace, this->enabledActions);
            return restrictToSafePart(safeStates, predList);
        }


        template<typename ValueType>
        storm::storage::BitVector WindowGame<ValueType>::directFWMP(storage::PredecessorsSquaredLinkedList<ValueType>& predList) {
            storm::storage::BitVector winGW = this->goodWin();
            if (winGW == this->restrictedStateSpace or winGW.empty()) {
                return winGW;
            }
            else {
                std::unique_ptr<WindowGame<ValueType>> safeGame = this->restrictToSafePart(winGW, predList);
                return safeGame->directFWMP(predList);
            }
        }

        template<typename ValueType>
        storm::storage::BitVector WindowMeanPayoffGame<ValueType>::goodWin() {
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
                    if (state == this->mdp.getNumberOfStates() - 1) {
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
        storm::storage::BitVector WindowParityGame<ValueType>::goodWin() {

        }

        template<typename ValueType>
        std::unique_ptr<WindowGame<ValueType>>
        WindowMeanPayoffGame<ValueType>::restrictToSafePart(
                storm::storage::BitVector const& safeStates,
                storage::PredecessorsSquaredLinkedList<ValueType> &predList) {

            storm::storage::BitVector badStates = (~safeStates) & this->restrictedStateSpace;
            storm::storage::BitVector restrictedStateSpace = this->restrictedStateSpace & safeStates;
            storm::storage::BitVector enabledActions = this->enabledActions;
            // disable all enabled actions of bad states
            for (uint_fast64_t state: badStates) {
                for (uint_fast64_t action = this->enabledActions.getNextSetIndex(this->matrix.getRowGroupIndices()[state]);
                     action < this->matrix.getRowGroupIndices()[state + 1];
                     action = this->enabledActions.getNextSetIndex(action + 1)) {
                    enabledActions.set(action, false);
                    predList.disableAction(action);
                }
            }

            // Initialize a stack to iterate on bad states and its (P2) attractors
            std::vector<uint_fast64_t> stack(badStates.begin(), badStates.end());
            uint_fast64_t currentBadState, state;
            while (!stack.empty()) {
                currentBadState = stack.back();
                stack.pop_back();
                for (const auto &action : predList.getStatePredecessors(currentBadState)) {
                    enabledActions.set(action, false);
                    predList.disableAction(action);
                    state = predList.getActionPredecessor(action);
                    if (! predList.hasEnabledActions(state)) {
                        restrictedStateSpace.set(state, false);
                        stack.push_back(state);
                    }
                }
            }

            return std::unique_ptr<WindowGame<ValueType>>(
                    new WindowMeanPayoffGame<ValueType>(this->mdp,
                                                        this->rewardModelName,
                                                        this->l_max,
                                                        restrictedStateSpace,
                                                        enabledActions));
        }

        template<typename ValueType>
        std::unique_ptr<WindowGame<ValueType>>
        WindowParityGame<ValueType>::restrictToSafePart(
                    storm::storage::BitVector const& safeStates,
                    storage::PredecessorsSquaredLinkedList<ValueType> &predList) {
            return std::unique_ptr<WindowGame<ValueType>>();
        }

        template class WindowGame<double>;
        template class WindowGame<storm::RationalNumber>;
        template class WindowMeanPayoffGame<double>;
        template class WindowMeanPayoffGame<storm::RationalNumber>;
        template class WindowParityGame<double>;

    }
}