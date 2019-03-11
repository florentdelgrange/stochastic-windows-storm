//
// Created by Florent Delgrange on 2019-02-01.
//

#include <storm/models/sparse/Mdp.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/storage/MaximalEndComponent.h>
#include <storm/storage/MaximalEndComponentDecomposition.h>
#include <stochastic-windows/game/WindowGame.h>

#ifndef STORM_MAXIMALENDCOMPONENTDECOMPOSITIONWINDOWGAME_H
#define STORM_MAXIMALENDCOMPONENTDECOMPOSITIONWINDOWGAME_H

namespace sw {
    namespace storage {

        template<typename ValueType>
        class MaximalEndComponentDecompositionWindowGame: public storm::storage::MaximalEndComponentDecomposition<ValueType> {
        public:

            /*!
             * This class is the maximal end component decomposition of the input MDP where each maximal end component can
             * be considered as a window game.
             *
             * @param mdp input model
             * @param rewardModelName name of the reward model for which the MDP will be considered as a game
             * @param l_max maximum window size
             */
            MaximalEndComponentDecompositionWindowGame(
                    storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max);

            /*!
             * Retrieves the window game of the input MEC
             * @param mec index of the MEC to consider as a game
             * @return the window game of the input MEC.
             */
            sw::game::WindowGame<ValueType> const& getGame(uint_fast64_t mec) const;

        protected:

            /*!
             * Generates the game version of each MEC
             */
            virtual void generateWindowGames(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max) = 0;

            std::vector<std::unique_ptr<sw::game::WindowGame<ValueType>>> windowGames;

        };

        template<typename ValueType>
        class MaximalEndComponentDecompositionWindowMeanPayoffGame: public MaximalEndComponentDecompositionWindowGame<ValueType> {
        public:

            MaximalEndComponentDecompositionWindowMeanPayoffGame(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max);

        protected:
            void generateWindowGames(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max) override;
        };

    }
}


#endif //STORM_MAXIMALENDCOMPONENTDECOMPOSITIONWINDOWGAME_H
