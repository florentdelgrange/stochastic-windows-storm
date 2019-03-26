//
// Created by Florent Delgrange on 2019-03-26.
//

#ifndef STORM_BOUNDEDWINDOWOBJECTIVE_H
#define STORM_BOUNDEDWINDOWOBJECTIVE_H

#include <stochastic-windows/WindowObjective.h>
#include <storm/storage/MaximalEndComponentDecomposition.h>
#include <storm/storage/BitVector.h>
#include <stochastic-windows/prefixindependent/MaximalEndComponentDecompositionWindowGame.h>
#include <stochastic-windows/prefixindependent/MaximalEndComponentDecompositionUnfolding.h>
#include <stochastic-windows/boundedwindow/MaximalEndComponentClassifier.cpp>
#include <storm/modelchecker/prctl/helper/SparseMdpPrctlHelper.h>
#include <storm/environment/Environment.h>

namespace sw {
    namespace BoundedWindow {

        template<typename ValueType>
        class BoundedWindowObjective: public WindowObjective<ValueType> {
        public:

            BoundedWindowObjective(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const& rewardModelName);

            /*!
             * Performs the MEC decomposition and classification of the input MDP and retrieves the good state space
             * for this bounded window objective that is the union of state spaces of good MECs.
             * @return a BitVector representing the good state space
             */
            virtual storm::storage::BitVector getGoodStateSpace() const = 0;

        };

        template<typename ValueType>
        class BoundedWindowMeanPayoffObjective: public BoundedWindowObjective<ValueType> {
        public:

            BoundedWindowMeanPayoffObjective(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const& rewardModelName);

            storm::storage::BitVector getGoodStateSpace() const override ;

        };

        template<typename ValueType>
        class BoundedWindowParityObjective: public BoundedWindowObjective<ValueType> {
        public:

            BoundedWindowParityObjective(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const& rewardModelName);

            storm::storage::BitVector getGoodStateSpace() const override ;

        };

        template<typename ValueType>
        std::vector<ValueType> performMaxProb(BoundedWindowObjective<ValueType> const &bwObjective);

        template<typename ValueType>
        ValueType performMaxProb(uint_fast64_t state, BoundedWindowObjective<ValueType> const &bwObjective);

    }

}

#endif //STORM_BOUNDEDWINDOWOBJECTIVE_H
