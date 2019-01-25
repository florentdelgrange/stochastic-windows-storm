//
// Created by Florent Delgrange on 2019-01-22.
//

#include <stochastic-windows/directfixedwindow/WindowUnfolding.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/models/sparse/StandardRewardModel.h>
#include <storm/utility/constants.h>
#include <storm/modelchecker/prctl/helper/SparseMdpPrctlHelper.h>
#include <storm/environment/Environment.h>
#include <storm/solver/SolveGoal.h>
#include <storm/storage/BitVector.h>
#include <storm/environment/solver/MinMaxSolverEnvironment.h>

#ifndef STORM_DIRECTFIXEDWINDOWOBJECTIVE_H
#define STORM_DIRECTFIXEDWINDOWOBJECTIVE_H

namespace sw {
    namespace DirectFixedWindow {

        template<typename ValueType>
        class DirectFixedWindowObjective {
        public:

            DirectFixedWindowObjective(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const& rewardModelName,
                    uint_fast64_t const& l_max);

            virtual std::unique_ptr<WindowUnfolding<ValueType>> performUnfolding(storm::storage::BitVector const& initialStates) const = 0;

            const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &getMdp() const;

            const std::string &getRewardModelName() const;

            uint_fast64_t getMaximumWindowSize();

        protected:

            storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp;
            std::string const& rewardModelName;
            uint_fast64_t const l_max;

        };

        template<typename ValueType>
        class DirectFixedWindowObjectiveMeanPayoff: public DirectFixedWindowObjective<ValueType> {
        public:

            DirectFixedWindowObjectiveMeanPayoff(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const& rewardModelName,
                    uint_fast64_t const& l_max);

            std::unique_ptr<WindowUnfolding<ValueType>> performUnfolding(storm::storage::BitVector const& initialStates) const override;

        };

        template<typename ValueType>
        class DirectFixedWindowObjectiveParity: public DirectFixedWindowObjective<ValueType> {
        public:

            DirectFixedWindowObjectiveParity(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const& rewardModelName,
                    uint_fast64_t const& l_max);

            std::unique_ptr<WindowUnfolding<ValueType>> performUnfolding(storm::storage::BitVector const& initialStates) const override;

        };

        template<typename ValueType>
        std::vector<ValueType> performMaxProb(storm::storage::BitVector const &phiStates,
                DirectFixedWindowObjective<ValueType> const &dfwObjective,
                bool useMecBasedTechnique = false);

        template<typename ValueType>
        ValueType performMaxProb(uint_fast64_t state,
                DirectFixedWindowObjective<ValueType> const &dfwObjective,
                bool useMecBasedTechnique = false);
    }
}


#endif //STORM_DIRECTFIXEDWINDOWOBJECTIVE_H
