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
    namespace FixedWindow {

        template<typename ValueType>
        class MECsUnfolding {
        public:

            MECsUnfolding(storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>>& mdp);

            /*!
             * Get the index of the MEC containing the input state. Note that 0 is a special value indicating that the
             * input state does not belong to any MEC
             */
            uint_fast64_t getMecIndex(uint_fast64_t state);

            storm::storage::MaximalEndComponentDecomposition<ValueType>& getMaximalEndComponentDecomposition();

            /*!
             * Get the index k of the MEC containing the input state as well as the index of the state
             * (state, currentSumOfWeights, currentWindowLength) in the kth matrix, being the matrix representing the
             * unfolding of the kth MEC.
             * Returns the special value 0 (the index of the sink state) if (state, currentSumOfWeights, currentWindowLength)
             * does not exist in the new matrix k.
             *
             * @param state state in the original matrix
             * @param currentSumOfWeights value of the current sum of weights in the window
             * @param currentWindowSize current window size
             */
            std::pair<uint_fast64_t, uint_fast64_t> getNewIndex(uint_fast64_t state,
                                                                ValueType currentSumOfWeights,
                                                                uint_fast64_t currentWindowSize);

            /*!
             * Returns the matrix representing the unfolding of the kth MEC of the original MDP.
             * Note that 0 is a special value and does not represent any MEC.
             */
            storm::storage::SparseMatrix<ValueType>& getUnfoldedMatrix(uint_fast64_t mec);

            /*!
             * Get the number of unfolded ECs.
             */
            uint_fast64_t getNumberOfUnfoldedECs();

            /*!
             * Get a vector containing, for each MEC k, the meaning of each state in the new matrix corresponding to the
             * unfolding of the MEC k, expressed as a tuple (s, w, l) where s (state) is the state in the original matrix,
             * w (currentSumOfWeights) is the current sum of weights in the unfolding and l (currentWindowLength) is
             * the current window length in the unfolding.
             *
             * @param k the index of the MEC containing the state for which the meaning is explained.
             */
            std::vector<sw::DirectFixedWindow::StateValueWindowSize<ValueType>> getNewStatesMeaning(uint_fast64_t k);

            /*!
             * Build the refined sub-MDP of the kth MEC corresponding to the unfolding of the kth MEC for the bound l_max.
             * Note that the MDP built has no label.
             * @param k the index of the MEC for which the unfolding has been built
             * @return the MDP of the unfolding of the kth MEC
             */
            std::shared_ptr<storm::models::sparse::Mdp<ValueType>> unfoldingAsMDP(uint_fast64_t k);

            /*!
             * Prints the unfolding of the given mec to the given output stream.
             * @param out The output stream
             * @param k The index of the unfolded MEC
             */
            void printToStream(std::ostream& out, uint_fast64_t k);


       protected:

            std::vector<sw::DirectFixedWindow::WindowUnfolding<ValueType>> unfoldedECs;

            /*!
             * Vector containing the index of the MEC of each state.
             * Note that 0 is a special value indicating that the state does not belong to any MEC.
             */
            std::vector<uint_fast64_t> mecIndices;

            storm::storage::MaximalEndComponentDecomposition<ValueType> mecDecomposition;

            void performMECDecomposition(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                    std::string const& rewardModelName,
                    uint_fast64_t const& l_max);

            virtual void unfoldEC(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &initialStates,
                    storm::storage::BitVector const &enabledActions);

       };

        template<typename ValueType>
        class MECsUnfoldingMeanPayoff: public MECsUnfolding<ValueType> {
        public:

            MECsUnfoldingMeanPayoff(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max);

        protected:

            void unfoldEC(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &initialStates,
                    storm::storage::BitVector const &enabledActions);
        };

        template<typename ValueType>
        class MECsUnfoldingParity: public MECsUnfolding<ValueType> {
        public:

            MECsUnfoldingParity(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max);

        protected:

            void unfoldEC(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &initialStates,
                    storm::storage::BitVector const &enabledActions);
        };

    }
}

#endif //STORM_ECSUNFOLDING_H
