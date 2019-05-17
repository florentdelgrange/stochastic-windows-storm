//
// Created by Florent Delgrange on 09/11/2018.
//

#include <iostream>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/storage/MaximalEndComponent.h>
#include <storm/storage/MaximalEndComponentDecomposition.h>
#include <storm/utility/builder.h>
#include <storm/storage/sparse/ModelComponents.h>
#include <storm/models/sparse/StateLabeling.h>
#include <stochastic-windows/directfixedwindow/WindowUnfolding.h>

#ifndef STORM_ECSUNFOLDING_H
#define STORM_ECSUNFOLDING_H

namespace sw {
    namespace storage {

        template<typename ValueType>
        class MaximalEndComponentDecompositionUnfolding: public storm::storage::MaximalEndComponentDecomposition<ValueType> {
        public:

            /*!
             * This class is the maximal end component decomposition of the input MDP where the unfolding of each
             * of these maximal end components for the maximal window size l_max and the input reward model is performed.
             *
             * @param mdp input model
             * @param rewardModelName name of the reward model for which the MDP will be unfold
             * @param l_max maximal window size
             */
            MaximalEndComponentDecompositionUnfolding(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max);

            /*!
             * Get the index of the MEC containing the input state.
             */
            uint_fast64_t getMecIndex(uint_fast64_t state);

            /*!
             * get the unfolding of the maximal end components of the input MDP
             */
            sw::DirectFixedWindow::WindowUnfolding<ValueType> const& getUnfolding() const;

            uint_fast64_t getMaximumWindowSize() const;


       protected:

            std::unique_ptr<sw::DirectFixedWindow::WindowUnfolding<ValueType>> unfoldedECs;
            uint_fast64_t l_max;

            /*!
             * Vector containing the index of the MEC of each state.
             * Note that 0 is a special value indicating that the state does not belong to any MEC.
             */
            std::vector<uint_fast64_t> mecIndices;

            void performMECsUnfolding(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max);

            virtual void unfold(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &initialStates,
                    storm::storage::BitVector const &enabledActions) = 0;

       };

        template<typename ValueType>
        class MaximalEndComponentDecompositionUnfoldingMeanPayoff: public MaximalEndComponentDecompositionUnfolding<ValueType> {
        public:

            MaximalEndComponentDecompositionUnfoldingMeanPayoff(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max);

        protected:

            void unfold(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &initialStates,
                    storm::storage::BitVector const &enabledActions) override;
        };

        template<typename ValueType>
        class MaximalEndComponentDecompositionUnfoldingParity: public MaximalEndComponentDecompositionUnfolding<ValueType> {
        public:

            MaximalEndComponentDecompositionUnfoldingParity(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max);

        protected:

            void unfold(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &initialStates,
                    storm::storage::BitVector const &enabledActions) override;
        };

    }
}

#endif //STORM_ECSUNFOLDING_H
