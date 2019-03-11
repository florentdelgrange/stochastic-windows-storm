//
// Created by Florent Delgrange on 2019-02-01.
//

#include "MaximalEndComponentClassifier.h"

namespace sw {
    namespace BoundedWindow {

        template <typename ValueType>
        MaximalEndComponentClassifier<ValueType>::MaximalEndComponentClassifier(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                sw::storage::MaximalEndComponentDecompositionWindowGame<ValueType> const& mecDecompositionGame)
                : sw::util::MaximalEndComponentClassifier<ValueType>(mdp, mecDecompositionGame) {

            for (uint_fast64_t k = 0; k < mecDecompositionGame.size(); ++ k) {
                sw::game::WindowGame<ValueType> const& mecGame = mecDecompositionGame.getGame(k);
                storm::storage::BitVector winningSet = mecGame.boundedProblem();
                // If the winning set in the current MEC game is not empty, then the MEC is classified as good.
                if (not winningSet.empty()) {
                    this->goodMECs.set(k, true);
                    // logical OR
                    this->goodStateSpace |= mecGame.getStateSpace();
                    this->safeStateSpace |= winningSet;
                }
            }
        }

        template class MaximalEndComponentClassifier<double>;
        template class MaximalEndComponentClassifier<storm::RationalNumber>;

    }
}