//
// Created by Florent Delgrange on 2019-02-12.
//

#include "MdpGame.h"

namespace sw {
    namespace game {

        template<typename ValueType>
        MdpGame<ValueType>::MdpGame(
                const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                storm::storage::BitVector const &restrictedStateSpace,
                storm::storage::BitVector const &enabledActions)
                : mdp(mdp),
                  matrix(mdp.getTransitionMatrix()),
                  restrictedStateSpace(restrictedStateSpace),
                  enabledActions(enabledActions) {

            STORM_LOG_ASSERT(restrictedStateSpace.size() == mdp.getNumberOfStates(),
                    "The size of the BitVector representing the state space of the Player 2 must be the number of states in the MDP");
            STORM_LOG_ASSERT(enabledActions.size() == this->matrix.getRowCount(),
                    "The size of the BitVector representing the state space of the Player 2 must be the number of actions in the system");
        }

        template<typename ValueType>
        MdpGame<ValueType>::~MdpGame() = default;

        template <typename ValueType>
        storm::storage::BitVector MdpGame<ValueType>::attractorsP1(storm::storage::BitVector const& targetSet,
                                                                   BackwardTransitions const& backwardTransitions) const {
            storm::storage::BitVector attractors = targetSet;
            storm::storage::BitVector actionVisited(this->enabledActions.size(), false);
            std::vector<uint_fast64_t> remainingActionSuccessors(this->enabledActions.size());

            // Initialize a stack to iterate on P1 attractors of the targetSet
            std::forward_list<uint_fast64_t> stack(attractors.begin(), attractors.end());

            uint_fast64_t currentState, state;
            while (!stack.empty()) {
                currentState = stack.front();
                stack.pop_front();
                // actions of backward transitions are assumed to be enabled
                for (const auto &action : backwardTransitions.statePredecessors[currentState]) {
                    state = backwardTransitions.actionPredecessor[action];
                    // looking at enabled actions of states not currently in the attractors of the target set
                    if (not attractors[state]) {
                        if (not actionVisited[action]) {
                            actionVisited.set(action, true);
                            // note that this only holds by the assumption that all successors of actions stay in the
                            // restricted state space
                            remainingActionSuccessors[action] = this->matrix.getRow(action).getNumberOfEntries();
                        }
                        --remainingActionSuccessors[action];
                        if (not remainingActionSuccessors[action]) {
                            // if just one action has all of its successors in the attractors, then its state-predecessor
                            // is also in the attractors and we don't have to check other of its actions
                            attractors.set(state, true);
                            stack.push_front(state);
                        }
                    }
                }
            }
            return attractors;
        }

        template <typename ValueType>
        GameStates MdpGame<ValueType>::attractorsP2(GameStates const& targetSet,
                                                    BackwardTransitions const& backwardTransitions) const {
            GameStates attractors = targetSet;
            storm::storage::BitVector stateVisited(this->restrictedStateSpace.size(), false);
            std::vector<uint_fast64_t> remainingEnabledActions(this->restrictedStateSpace.size());

            // Initialize a stack to iterate on P2 attractors of the targetSet
            std::forward_list<uint_fast64_t> stack(attractors.p1States.begin(), attractors.p1States.end());

            // Visit the input state and decrement the number of remaining enabled actions to visit belonging to this state.
            // If all of the enabled actions of this state have been visited, then it belongs to the attractors.
            auto visitState = [&](uint_fast64_t state) -> void {
                if (not stateVisited[state]) {
                    stateVisited.set(state, true);
                    remainingEnabledActions[state] = backwardTransitions.numberOfEnabledActions[state];
                }
                -- remainingEnabledActions[state];
                if (not remainingEnabledActions[state]) {
                    attractors.p1States.set(state, true);
                    stack.push_front(state);
                }
            };

            // fill in the init stack by looking at predecessors of actions of the P2 attractors
            uint_fast64_t state;
            for (uint_fast64_t action: attractors.p2States) {
                state = backwardTransitions.actionPredecessor[action];
                if (not attractors.p1States[state]) {
                    visitState(state);
                }
            }

            uint_fast64_t currentState;
            while (not stack.empty()) {
                currentState = stack.front();
                stack.pop_front();
                for (uint_fast64_t action: backwardTransitions.statePredecessors[currentState]) {
                    if (not attractors.p2States[action]) {
                        attractors.p2States.set(action, true);
                        // the state for which the current action belongs to is not in the attractors, otherwise,
                        // the current action would be in it
                        state = backwardTransitions.actionPredecessor[action];
                        visitState(state);
                    }
                }
            }

            return attractors;
        }

        template<typename ValueType>
        void MdpGame<ValueType>::initBackwardTransitions(BackwardTransitions &backwardTransitions) const {
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
                        // note that here, this successor state is assumed to be in the restricted state space
                        backwardTransitions.statePredecessors[successorState].push_front(action);
                    }
                }
            }
        }

        template class MdpGame<double>;
        template class MdpGame<storm::RationalNumber>;

    }
}