//
// Created by Florent Delgrange on 2019-02-01.
//

#ifndef STORM_MAXIMALENDCOMPONENTCLASSIFIER_H
#define STORM_MAXIMALENDCOMPONENTCLASSIFIER_H

#include <storm/storage/MaximalEndComponent.h>
#include <stochastic-windows/game/WindowGame.h>
#include <stochastic-windows/fixedwindow/MaximalEndComponentDecompositionUnfolding.h>
#include <stochastic-windows/fixedwindow/MaximalEndComponentDecompositionWindowGame.h>
#include <storm/utility/graph.h>

namespace sw {
    namespace FixedWindow {

        template<typename ValueType>
        class MaximalEndComponentClassifier {
        public:

            /*!
             * This class allows to classify maximal end components as being good or not.
             *
             * @param mdp input model
             * @param mecDecompositionUnfolding unfolding of each maximal end component of the input model
             */
            MaximalEndComponentClassifier(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType> const& mecDecompositionUnfolding);

            /*!
             * This class allows to classify maximal end components as being good or not.
             *
             * @param mdp input model
             * @param mecDecompositionUnfolding window-game version of each maximal end component of the input model
             */
            MaximalEndComponentClassifier(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    sw::storage::MaximalEndComponentDecompositionWindowGame<ValueType> const& mecDecompositionGame);

            /*!
             * Retrieves a vector containing each good maximal end components.
             */
            std::vector<std::reference_wrapper<const storm::storage::MaximalEndComponent>> getGoodMaximalEndComponents();
            /*!
             * Retrieves the set of safe states of the input model.
             */
            storm::storage::BitVector const& getSafeStateSpace();
            /*!
             * Retrieves the set of good states of the input model (i.e., the union of the state space of eacg good MEC).
             */
            storm::storage::BitVector const& getGoodStateSpace();

        private:

            storm::storage::MaximalEndComponentDecomposition<ValueType> const& maximalEndComponentDecomposition;
            storm::storage::BitVector safeStateSpace;
            storm::storage::BitVector goodStateSpace;
            storm::storage::BitVector goodMECs;

        };

    }
}


#endif //STORM_MAXIMALENDCOMPONENTCLASSIFIER_H
