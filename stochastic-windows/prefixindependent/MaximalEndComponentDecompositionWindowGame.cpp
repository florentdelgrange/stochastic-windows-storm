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
                : storm::storage::MaximalEndComponentDecomposition<ValueType>(mdp),
                  mecIndices(mdp.getNumberOfStates()),
                  l_max(l_max) {
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
        MaximalEndComponentDecompositionWindowMeanPayoffGame<ValueType>::MaximalEndComponentDecompositionWindowMeanPayoffGame(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const& rewardModelName)
                : MaximalEndComponentDecompositionWindowGame<ValueType>::MaximalEndComponentDecompositionWindowGame(mdp, rewardModelName, 0) {
            this->generateWindowGames(mdp, rewardModelName, 0);
        }

        template <typename ValueType>
        MaximalEndComponentDecompositionWindowParityGame<ValueType>::MaximalEndComponentDecompositionWindowParityGame(
                const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                std::string const &rewardModelName)
                : MaximalEndComponentDecompositionWindowGame<ValueType>::MaximalEndComponentDecompositionWindowGame(mdp, rewardModelName, 0) {
            this->generateWindowGames(mdp, rewardModelName, 0);
        }

        template <typename ValueType>
        uint_fast64_t sw::storage::MaximalEndComponentDecompositionWindowGame<ValueType>::getMecIndex(uint_fast64_t state) const {
            STORM_LOG_ASSERT(mecIndices[state] != 0, "The state " << state << " does not belong to any MEC.");
            return mecIndices[state] - 1;
        }

        template<typename ValueType>
        uint_fast64_t MaximalEndComponentDecompositionWindowGame<ValueType>::getMaximumWindowSize() const {
            return this->l_max;
        }

        template<typename ValueType>
        void MaximalEndComponentDecompositionWindowMeanPayoffGame<ValueType>::generateWindowGames(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const& rewardModelName,
                uint_fast64_t const& l_max) {

            uint_fast64_t k = 0;
            for (storm::storage::MaximalEndComponent const &mec: *this) {
                storm::storage::BitVector restrictedStateSpace(mdp.getNumberOfStates(), false);
                storm::storage::BitVector enabledActions(mdp.getNumberOfChoices(), false);
                ++ k;
                for (auto state: mec.getStateSet()){
                    restrictedStateSpace.set(state, true);
                    this->mecIndices[state] = k;
                    for (auto action: mec.getChoicesForState(state)) {
                        enabledActions.set(action, true);
                    }
                }

                if (l_max) {
                    this->windowGames.push_back(
                            std::unique_ptr<sw::game::WindowGame<ValueType>>(
                                    new sw::game::WindowMeanPayoffGame<ValueType>(mdp, rewardModelName, l_max, std::move(restrictedStateSpace), std::move(enabledActions))
                            )
                    );
                }
                else {
                    this->windowGames.push_back(
                            std::unique_ptr<sw::game::WindowGame<ValueType>>(
                                    new sw::game::WindowMeanPayoffGame<ValueType>(mdp, rewardModelName, std::move(restrictedStateSpace), std::move(enabledActions))
                            )
                    );
                }
            }
        }

        template<typename ValueType>
        void MaximalEndComponentDecompositionWindowParityGame<ValueType>::generateWindowGames(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const& rewardModelName,
                uint_fast64_t const& l_max) {

            uint_fast64_t k = 0;
            for (storm::storage::MaximalEndComponent const &mec: *this) {
                storm::storage::BitVector restrictedStateSpace(mdp.getNumberOfStates(), false);
                storm::storage::BitVector enabledActions(mdp.getNumberOfChoices(), false);
                ++ k;
                for (auto state: mec.getStateSet()){
                    restrictedStateSpace.set(state, true);
                    this->mecIndices[state] = k;
                    for (auto action: mec.getChoicesForState(state)) {
                        enabledActions.set(action, true);
                    }
                }

                this->windowGames.push_back(
                        std::unique_ptr<sw::game::WindowGame<ValueType>>(
                                new sw::game::WindowParityGame<ValueType>(mdp, rewardModelName, std::move(restrictedStateSpace), std::move(enabledActions))
                        )
                );
            }
        }

        template<typename ValueType>
        sw::game::WindowGame<ValueType> const& MaximalEndComponentDecompositionWindowGame<ValueType>::getGame(uint_fast64_t mec) const {
            return *(this->windowGames[mec]);
        }

        template class  MaximalEndComponentDecompositionWindowGame<double>;
        template class  MaximalEndComponentDecompositionWindowGame<storm::RationalNumber>;
        template class  MaximalEndComponentDecompositionWindowMeanPayoffGame<double>;
        template class  MaximalEndComponentDecompositionWindowMeanPayoffGame<storm::RationalNumber>;
        template class  MaximalEndComponentDecompositionWindowParityGame<double>;
        template class  MaximalEndComponentDecompositionWindowParityGame<storm::RationalNumber>;

    }
}
