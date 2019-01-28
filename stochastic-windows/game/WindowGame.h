//
// Created by Florent Delgrange on 2019-01-25.
//

#include <storm/storage/BitVector.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/SparseMatrix.h>

#ifndef STORM_WINDOWGAME_H
#define STORM_WINDOWGAME_H

namespace sw {
    namespace Game {

        template<typename ValueType>
        class WindowGame {
        public:

            /**
             * Consider the MDP as a game to synthesize strategies for window objectives.
             *
             * @param mdp the model to consider as a game
             * @param rewardModelName the name of the reward model to consider
             * @param l_max the maximum window size
             * @param restrictedStateSpace the set of states to consider in the MDP-game
             * @param enabledActions the set of actions to consider in the MDP-game
             */
            WindowGame(
                    storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &restrictedStateSpace,
                    storm::storage::BitVector const &enabledActions);

            /*
             * Make destructor virtual to allow deleting objects through pointer to base class(es).
             */
            virtual ~WindowGame() {}

            /**
             * Compute the winning set of states from which there exists a strategy allowing to surely close a window in
             * l_max steps or less
             *
             * @return the winning set for the GoodWindow objective
             */
            virtual storm::storage::BitVector goodWin() = 0;

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
             * Reward Model of the MDP
             */
            storm::models::sparse::StandardRewardModel <ValueType> const &rewardModel;
            /**
             * maximum window size
             */
            uint_fast64_t const &l_max;
            /**
             * set of actions to consider in the MDP-game
             */
            storm::storage::BitVector const &restrictedStateSpace;
            /**
             * set of actions to consider in the MDP-game
             */
            storm::storage::BitVector const &enabledActions;

            /**
             * Compute the set of successor states of each (state, action) pair w.r.t. the restricted state space and
             * enabled actions considered.
             */
            std::vector<storm::storage::BitVector> getSuccessorStates();

        };

        template<typename ValueType>
        class WindowMeanPayoffGame : public WindowGame<ValueType> {
        public:

            /**
             * Consider the MDP as a game to synthesize strategies for window mean-payoff objectives.
             *
             * @param mdp the model to consider as a game
             * @param rewardModelName the name of the reward model to consider
             * @param l_max the maximum window size
             * @param restrictedStateSpace the set of states to consider in the MDP-game
             * @param enabledActions the set of actions to consider in the MDP-game
             */
            WindowMeanPayoffGame(
                    storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &restrictedStateSpace,
                    storm::storage::BitVector const &enabledActions);

            storm::storage::BitVector goodWin() override;

        };

        template<typename ValueType>
        class WindowParityGame : public WindowGame<ValueType> {
        public:

            /**
             * Consider the MDP as a game to synthesize strategies for window parity objectives.
             *
             * @param mdp the model to consider as a game
             * @param rewardModelName the name of the reward model to consider
             * @param l_max the maximum window size
             * @param restrictedStateSpace the set of states to consider in the MDP-game
             * @param enabledActions the set of actions to consider in the MDP-game
             */
            WindowParityGame(
                    storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &restrictedStateSpace,
                    storm::storage::BitVector const &enabledActions);

            storm::storage::BitVector goodWin() override;

        };
    }
}


#endif //STORM_WINDOWGAME_H
