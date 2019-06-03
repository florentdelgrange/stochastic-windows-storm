//
// Created by Florent Delgrange on 2019-03-21.
//

#include "WeakParityGame.h"

namespace sw {
    namespace game {


        template<typename ValueType>
        WeakParityGame<ValueType>::WeakParityGame(
                const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                std::string const &rewardModelName, storm::storage::BitVector const &restrictedStateSpace,
                storm::storage::BitVector const &enabledActions)
                : MdpGame<ValueType>(mdp, restrictedStateSpace, enabledActions),
                  rewardModelName(rewardModelName),
                  rewardModel(mdp.getRewardModel(rewardModelName)) {}

        template<typename ValueType>
        typename WeakParityGame<ValueType>::WinningRegion WeakParityGame<ValueType>::weakParity() const {
            // winning set initialization
            WinningRegion W;
            W.winningSetP1.p1States = storm::storage::BitVector(this->restrictedStateSpace.size(), false);
            W.winningSetP1.p2States = storm::storage::BitVector(this->enabledActions.size(), false);
            W.winningSetP2.p1States = storm::storage::BitVector(this->restrictedStateSpace.size(), false);
            W.winningSetP2.p2States = storm::storage::BitVector(this->enabledActions.size(), false);

            // priorities handling
            std::vector<ValueType> const& priorities = this->rewardModel.getStateRewardVector();
            std::map<ValueType, storm::storage::BitVector, storm::utility::ElementLess<ValueType>> priorityToStatesMapping;
            // map each priority with the set of states having this priority
            for (uint_fast64_t state: this->restrictedStateSpace) {
                ValueType priority = priorities[state];
                auto it = priorityToStatesMapping.find(priority);
                if (it != priorityToStatesMapping.end()) {
                    storm::storage::BitVector &states = it->second;
                    states.set(state, true);
                }
                else {
                    storm::storage::BitVector states(priorities.size(), false);
                    states.set(state, true);
                    priorityToStatesMapping.insert(std::make_pair(priority, states));
                }
            }

            storm::storage::BitVector currentStateSpace = this->restrictedStateSpace;
            storm::storage::BitVector currentActionSpace = this->enabledActions;
            for(typename std::map<ValueType, storm::storage::BitVector>::iterator iter = priorityToStatesMapping.begin();
                iter != priorityToStatesMapping.end(); ++iter) {

                ValueType priority = iter->first;
                GameStates priorityAttractors, T;
                T.p1States = currentStateSpace & iter->second;
                T.p2States = storm::storage::BitVector(this->enabledActions.size(), false);
                BackwardTransitions backwardTransitions;
                WeakParityGame restrictedGame(this->mdp, this->rewardModelName, currentStateSpace, currentActionSpace);
                restrictedGame.initBackwardTransitions(backwardTransitions);
                if (isEven(priority)) {
                    priorityAttractors = restrictedGame.attractorsP1(T, backwardTransitions);
                    W.winningSetP1.p1States |= priorityAttractors.p1States;
                    W.winningSetP1.p2States |= priorityAttractors.p2States;
                }
                else {
                    priorityAttractors = restrictedGame.attractorsP2(T, backwardTransitions);
                    W.winningSetP2.p1States |= priorityAttractors.p1States;
                    W.winningSetP2.p2States |= priorityAttractors.p2States;
                }
                currentStateSpace &= ~priorityAttractors.p1States;
                currentActionSpace &= ~priorityAttractors.p2States;
                if (currentStateSpace.empty()) {
                    break;
                }
            }
            return W;
        }

        template <typename ValueType>
        bool WeakParityGame<ValueType>::isEven(ValueType priority) const {
            return storm::utility::isZero(storm::utility::mod<ValueType>(priority, 2));
        }

        template<typename ValueType>
        void WeakParityGame<ValueType>::initBackwardTransitions(BackwardTransitions &backwardTransitions) const {
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
                        // in weak parity games, actions may lead to states not belonging to the restricted state space
                        if (this->restrictedStateSpace[successorState]) {
                            backwardTransitions.statePredecessors[successorState].push_front(action);
                        }
                    }
                }
            }
        }

        template class WeakParityGame<double>;
    }

}