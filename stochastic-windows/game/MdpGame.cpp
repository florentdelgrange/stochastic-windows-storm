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

        template class MdpGame<double>;
        template class MdpGame<storm::RationalNumber>;

    }
}