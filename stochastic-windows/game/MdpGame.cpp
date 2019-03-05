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
            std::vector<uint_fast64_t> remainingActionsSuccessors(this->enabledActions.size());

            // Initialize a stack to iterate on P1 attractors of the targetSet
            std::forward_list<uint_fast64_t> stack(attractors.begin(), attractors.end());

            uint_fast64_t currentState, state;
            while (!stack.empty()) {
                currentState = stack.front();
                stack.pop_front();
                for (const auto &action : backwardTransitions.statesPredecessors[currentState]) {
                    state = backwardTransitions.actionsPredecessor[action];
                    // looking at enabled actions of states not currently in the attractors of the target set
                    if (this->enabledActions[action] and not attractors[state]) {
                        if (not actionVisited[action]) {
                            actionVisited.set(action, true);
                            // note that this only holds by the assumption all successors of actions staying in the
                            // restricted state space
                            remainingActionsSuccessors[action] = this->matrix.getRow(action).getNumberOfEntries();
                        }
                        remainingActionsSuccessors[action]--;
                        if (not remainingActionsSuccessors[action]) {
                            // if just one action has all of its successors in the attractors, then its state-predecessor
                            // is also in the attrators and we don't have to check other of its actions
                            attractors.set(state, true);
                            stack.push_front(state);
                        }
                    }
                }
            }
            return attractors;
        }

        template <typename ValueType>
        GameStates MdpGame<ValueType>::attractorsP2(storm::storage::BitVector const& targetSet,
                                                    BackwardTransitions const& backwardTransitions) const {

        }

        template class MdpGame<double>;
        template class MdpGame<storm::RationalNumber>;

    }
}