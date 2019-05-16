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
             * Get the index k of the MEC containing the input state as well as the index of the state
             * (state, currentSumOfWeights, currentWindowLength) in the kth matrix, being the matrix representing the
             * unfolding of the kth MEC.
             *
             * @param state state in the original matrix
             * @param currentSumOfWeights value of the current sum of weights in the window
             * @param currentWindowSize current window size
             */
            std::pair<uint_fast64_t, uint_fast64_t> getNewIndex(uint_fast64_t state,
                                                                ValueType currentSumOfWeights,
                                                                uint_fast64_t currentWindowSize);

            /*!
             * Retrieves the matrix representing the unfolding of the kth MEC of the original MDP.
             */
            storm::storage::SparseMatrix<ValueType> const& getUnfoldedMatrix(uint_fast64_t mec);
            storm::storage::SparseMatrix<ValueType> const& getUnfoldedMatrix(uint_fast64_t mec) const;

            /*!
             * Retrieves a vector containing, for each MEC k, the meaning of each state in the new matrix corresponding to the
             * unfolding of the MEC k, expressed as a tuple (s, w, l) where s (state) is the state in the original matrix,
             * w (currentSumOfWeights) is the current sum of weights in the unfolding and l (currentWindowLength) is
             * the current window length in the unfolding.
             *
             * @param k the index of the MEC containing the state for which the meaning is explained.
             */
            std::vector<sw::DirectFixedWindow::StateValueWindowSize<ValueType>> getNewStatesMeaning(uint_fast64_t k);

            /*!
             * Builds the refined sub-MDP of the kth MEC corresponding to the unfolding of the kth MEC for the bound l_max.
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

            /**
             * Retrieves the index in the kth unfolding of the input initial state.
             * @param k index of the MEC containing the state representing the input initial state
             * @param initialState
             * @return the index of the input initial state in the unfolding if it exists, 0 else
             */
            uint_fast64_t getInitialState(uint_fast64_t k, uint_fast64_t initialState);
            uint_fast64_t getInitialState(uint_fast64_t k, uint_fast64_t initialState) const;
            uint_fast64_t getMaximumWindowSize() const;


       protected:

            std::vector<std::unique_ptr<sw::DirectFixedWindow::WindowUnfolding<ValueType>>> unfoldedECs;
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

            virtual void unfoldEC(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
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

            void unfoldEC(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
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

            void unfoldEC(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &initialStates,
                    storm::storage::BitVector const &enabledActions) override;
        };

    }
}

#endif //STORM_ECSUNFOLDING_H
