//
// Created by Florent Delgrange on 09/11/2018.
//

#include <iostream>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/SparseMatrix.h>

#ifndef STORM_BNDGOODWINDOWMP_H
#define STORM_BNDGOODWINDOWMP_H

#endif //STORM_BNDGOODWINDOWMP_H

namespace sw {
    namespace BndGoodWindowMP {
        template<typename ValueType>
        class ECsUnfolding {
        public:

            /**
             * Unfold the end components of the mdp entered in parameter for the window mean payoff problem w.r.t. the
             * maximal windows' length l_max.
             *
             * @param mdp Markov decision process for which the ECs will be unfolded
             * @param rewardModelName name of the reward model following which the mdp will be unfolded
             * @param l_max the maximum length of windows to consider
             */
            ECsUnfolding(std::shared_ptr<storm::models::sparse::Mdp<ValueType>> mdp,
                         std::string const& rewardModelName,
                         uint_fast64_t const &l_max);
            /**
             * Get the index of the MEC containing the input state. Note that 0 is a special value indicating that the
             * input state does not belong to any MEC
             */
            uint_fast64_t getMecIndex(uint_fast64_t state);
            /**
             * Get the index k of the MEC containing the input state as well as the index of the state
             * (state, currentSumOfWeights, currentWindowLength) in the kth matrix, being the matrix representing the
             * unfolding of the kth MEC.
             * Returns the special value 0 (the index of the sink state) if (state, currentSumOfWeights, currentWindowLength)
             * does not exist in the new matrix k.
             *
             * @param state state in the original matrix
             * @param currentSumOfWeights value of the current sum of weights in the window
             * @param currentWindowLength value of the current window length
             */
            std::pair<uint_fast64_t, uint_fast64_t> getNewIndex(uint_fast64_t state, ValueType currentSumOfWeights,
                                                                uint_fast64_t currentWindowLength);
            /**
             * Returns the matrix representing the unfolding of the kth MEC of the original MDP.
             * Note that 0 is a special value and does not represent any MEC.
             */
            storm::storage::SparseMatrix<ValueType> getUnfoldedMatrix(uint_fast64_t mec);

        private:

            /**
             * For each MEC in the MEC decomposition, this matrix represents the unfolded MEC for the window MP
             * objective.
             */
            std::vector<storm::storage::SparseMatrix<ValueType>> matrices;
            /**
             * Vector containing the index of the MEC of each state.
             * Note that 0 is a special value indicating that the state does not belong to any MEC.
             */
            std::vector<uint_fast64_t> mecIndices;
            std::vector<std::vector<std::unordered_map<ValueType, uint_fast64_t>>> windowVector;
            /**
             * For each MEC of the original MDP, this vector maps each new state to its pairs of
             * (successor states, probabilities) for each of its enabled actions that are ordered according to the
             * ordering of enabled actions of each state of the MEC.
             * Dimensions = (0: MEC, 1: state, 2: action, 3: pair of (s', p) where p is the probability to go to s').
             */
            std::vector<std::vector<std::vector<std::pair<uint_fast64_t, ValueType>>>> newRowGroupEntries;
            /**
             * Unfold a MEC from a given state. The result fields oldToNewStateMapping and newRowGroupEntries are
             * filled accordingly.
             *
             * @param state
             * @param currentSumOfWeights
             * @param l
             * @param l_max
             * @param originalMatrix
             * @param rewardModel
             * @param currentMec
             * @return the index of the input state in the new matrix of the current MEC
             */
            uint_fast64_t unfoldFrom(u_int64_t const& state, ValueType const& currentSumOfWeights, uint_fast64_t const& l,
                                     uint_fast64_t const& l_max,
                                     storm::storage::SparseMatrix<ValueType> const &originalMatrix,
                                     std::vector<ValueType> const& stateActionRewardsVector,
                                     storm::storage::MaximalEndComponent const &currentMec);
        };
    }
}