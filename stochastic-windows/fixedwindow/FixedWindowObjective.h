//
// Created by Florent Delgrange on 2019-02-05.
//

#ifndef STORM_FIXEDWINDOWOBJECTIVE_H
#define STORM_FIXEDWINDOWOBJECTIVE_H

#include <stochastic-windows/WindowObjective.h>
#include <storm/storage/MaximalEndComponentDecomposition.h>
#include <storm/storage/BitVector.h>
#include <stochastic-windows/prefixindependent/MaximalEndComponentDecompositionWindowGame.h>
#include <stochastic-windows/prefixindependent/MaximalEndComponentDecompositionUnfolding.h>
#include <stochastic-windows/fixedwindow/MaximalEndComponentClassifier.cpp>
#include <storm/modelchecker/prctl/helper/SparseMdpPrctlHelper.h>
#include <storm/environment/Environment.h>

namespace sw {
    namespace FixedWindow {

        template<typename ValueType>
        class FixedWindowObjective: public WindowObjective<ValueType> {
        public:

            FixedWindowObjective(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const& rewardModelName,
                    uint_fast64_t const& l_max);

            uint_fast64_t getMaximumWindowSize() const;
            /*!
             * Performs the MEC decomposition and classification of the input MDP and retrieves the good state space
             * for this fixed window objective that is the union of state spaces of good MECs.
             * @return a BitVector representing the good state space
             */
            virtual storm::storage::BitVector getGoodStateSpace() const = 0;
            virtual sw::storage::GoodStateSpaceAndScheduler<ValueType> produceGoodScheduler() const = 0;

        protected:

            uint_fast64_t const l_max;

        };

        template<typename ValueType>
        class FixedWindowMeanPayoffObjective: public FixedWindowObjective<ValueType> {
        public:

            FixedWindowMeanPayoffObjective(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const& rewardModelName,
                    uint_fast64_t const& l_max,
                    bool windowGameBasedClassification = true);

            storm::storage::BitVector getGoodStateSpace() const override ;
            sw::storage::GoodStateSpaceAndScheduler<ValueType> produceGoodScheduler() const override;

        private:

            bool windowGameClassification;

        };

        template<typename ValueType>
        class FixedWindowParityObjective: public FixedWindowObjective<ValueType> {
        public:

            FixedWindowParityObjective(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const& rewardModelName,
                    uint_fast64_t const& l_max);

            storm::storage::BitVector getGoodStateSpace() const override ;
            sw::storage::GoodStateSpaceAndScheduler<ValueType> produceGoodScheduler() const override;

        };

        template<typename ValueType>
        sw::storage::ValuesAndScheduler<ValueType> performMaxProb(FixedWindowObjective<ValueType> const &fwObjective, bool produceScheduler = false);

    }
}


#endif //STORM_FIXEDWINDOWOBJECTIVE_H
