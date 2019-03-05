//
// Created by Florent Delgrange on 2019-02-12.
//

#ifndef STORM_MDPGAME_H
#define STORM_MDPGAME_H

#include <storm/models/sparse/StandardRewardModel.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/BitVector.h>

namespace sw {
    namespace game {

        struct BackwardTransitions {
            std::vector<std::forward_list<uint_fast64_t>> statesPredecessors;
            std::vector<uint_fast64_t> actionsPredecessor;
            std::vector<uint_fast64_t> numberOfEnabledActions;
        };

        struct GameStates {
            storm::storage::BitVector p1States;
            storm::storage::BitVector p2States;
        };

        template<typename ValueType>
        class MdpGame {
        public:
            /*!
             * Consider the input MDP as a game to verify that objectives are surely verified in the system.
             *
             * @tparam ValueType
             * @param mdp
             * @param restrictedStateSpace the set of states to consider in the MDP-game forming the state space of the
             *        Player 1 in the MDP-game.
             * @param enabledActions the set of actions to consider in the MDP-game forming the state space of the
             *        Player 2 in the MDP-game.
             * @note strong assumption: choosing an enabled action must always lead to a state of the restricted state space.
             */
            MdpGame(storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    storm::storage::BitVector const &restrictedStateSpace,
                    storm::storage::BitVector const &enabledActions);

            virtual ~MdpGame() = 0;

            /*!
             * compute the P1 attractors of the target set in this MDP game
             * @param targetSet set of states of the MDP
             * @param backwardTransitions backward transitions of the MDP game
             * @return the P1 attractors of the target set in this MDP game
             */
            storm::storage::BitVector attractorsP1(storm::storage::BitVector const& targetSet,
                                                   BackwardTransitions const& backwardTransitions) const;

            GameStates attractorsP2(storm::storage::BitVector const& targetSet,
                                    BackwardTransitions const& backwardTransitions) const;

        protected:
            /**
             * MDP to consider as a game
             */
            storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp;
            /**
             * matrix of the MDP to consider as a game
             */
            storm::storage::SparseMatrix<ValueType> const &matrix;
            /**
             * set of states to consider in the MDP-game
             */
            storm::storage::BitVector const restrictedStateSpace;
            /**
             * set of actions to consider in the MDP-game
             */
            storm::storage::BitVector const enabledActions;
        };

    }
}


#endif //STORM_MDPGAME_H
