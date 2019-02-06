//
// Created by Florent Delgrange on 2019-02-01.
//

#include "MaximalEndComponentDecompositionWindowGame.h"

namespace sw {
    namespace storage {

        template<typename ValueType>
        MaximalEndComponentDecompositionWindowGame<ValueType>::MaximalEndComponentDecompositionWindowGame(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                std::string const &rewardModelName, uint_fast64_t const &l_max)
                : storm::storage::MaximalEndComponentDecomposition<ValueType>(mdp) {
            this->windowGames.reserve(this->size());
        }

        template<typename ValueType>
        MaximalEndComponentDecompositionWindowMeanPayoffGame<ValueType>::MaximalEndComponentDecompositionWindowMeanPayoffGame(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const& rewardModelName, uint_fast64_t const& l_max)
                : MaximalEndComponentDecompositionWindowGame<ValueType>::MaximalEndComponentDecompositionWindowGame(mdp, rewardModelName, l_max) {
            this->generateWindowGames(mdp, rewardModelName, l_max);
        }

        template<typename ValueType>
        void MaximalEndComponentDecompositionWindowMeanPayoffGame<ValueType>::generateWindowGames(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const& rewardModelName,
                uint_fast64_t const& l_max) {

            for (storm::storage::MaximalEndComponent const &mec: *this) {
                storm::storage::BitVector restrictedStateSpace(mdp.getNumberOfStates(), false);
                storm::storage::BitVector enabledActions(mdp.getNumberOfChoices(), false);

                for (auto state: mec.getStateSet()){
                    restrictedStateSpace.set(state, true);
                    for (auto action: mec.getChoicesForState(state)) {
                        enabledActions.set(action, true);
                    }
                }

                this->windowGames.push_back(
                        std::unique_ptr<sw::Game::WindowGame<ValueType>>(
                                new sw::Game::WindowMeanPayoffGame<ValueType>(mdp, rewardModelName, l_max, std::move(restrictedStateSpace), std::move(enabledActions))
                                )
                        );
            }
        }

        template<typename ValueType>
        sw::Game::WindowGame<ValueType> const& MaximalEndComponentDecompositionWindowGame<ValueType>::getGame(uint_fast64_t mec) const {
            return *(this->windowGames[mec]);
        }

        template class  MaximalEndComponentDecompositionWindowGame<double>;
        template class  MaximalEndComponentDecompositionWindowGame<storm::RationalNumber>;
        template class  MaximalEndComponentDecompositionWindowMeanPayoffGame<double>;
        template class  MaximalEndComponentDecompositionWindowMeanPayoffGame<storm::RationalNumber>;

    }
}
