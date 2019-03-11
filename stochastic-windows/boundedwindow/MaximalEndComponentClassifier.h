//
// Created by Florent Delgrange on 2019-02-01.
//

#ifndef STORM_FW_MAXIMALENDCOMPONENTCLASSIFIER_H
#define STORM_FW_MAXIMALENDCOMPONENTCLASSIFIER_H

#include <storm/storage/MaximalEndComponent.h>
#include <stochastic-windows/game/WindowGame.h>
#include <stochastic-windows/prefixindependent/MaximalEndComponentDecompositionUnfolding.h>
#include <stochastic-windows/prefixindependent/MaximalEndComponentDecompositionWindowGame.h>
#include <stochastic-windows/prefixindependent/MaximalEndComponentClassifier.h>
#include <storm/utility/graph.h>

namespace sw {
    namespace BoundedWindow {

        template<typename ValueType>
        class MaximalEndComponentClassifier: public sw::util::MaximalEndComponentClassifier<ValueType> {
        public:

            /*!
             * This class allows to classify maximal end components as being good or not for the Bounded Window objective.
             *
             * @note A good EC (end component) w.r.t. a fixed window objective is an EC for which there exists a
             *       sub-EC where there exists a strategy surely winning this fixed window objective from every of its
             *       state.
             * @param mdp input model
             * @param mecDecompositionUnfolding window-game version of each maximal end component of the input model
             */
            MaximalEndComponentClassifier(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    sw::storage::MaximalEndComponentDecompositionWindowGame<ValueType> const& mecDecompositionGame);

        };

    }
}


#endif //STORM_FW_MAXIMALENDCOMPONENTCLASSIFIER_H
