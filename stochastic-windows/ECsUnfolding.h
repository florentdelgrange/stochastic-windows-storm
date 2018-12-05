//
// Created by Florent Delgrange on 09/11/2018.
//

#include <iostream>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/storage/MaximalEndComponent.h>
#include <storm/storage/MaximalEndComponentDecomposition.h>
#include <storm/models/sparse/Mdp.h>

#ifndef STORM_ECSUNFOLDING_H
#define STORM_ECSUNFOLDING_H

namespace sw {
    namespace WindowMP {

        template<typename ValueType>
        struct StateWeightWindowLength{
            uint_fast64_t state;
            ValueType currentSumOfWeights;
            uint_fast64_t currentWindowLength;
        };

        template<typename ValueType>
        class ECsUnfolding {
        public:

            /*!
             * Constructs an unfolding of the end components of the mdp entered in parameter for the window mean payoff
             * problem w.r.t. the maximal windows' length l_max.
             *
             * @param mdp Markov decision process for which the ECs will be unfolded
             * @param rewardModelName name of the reward model following which the mdp will be unfolded
             * @param l_max the maximum length of windows to consider
             */
            ECsUnfolding(storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                         std::string const& rewardModelName,
                         uint_fast64_t const& l_max);

            /*!
             * Get the index of the MEC containing the input state. Note that 0 is a special value indicating that the
             * input state does not belong to any MEC
             */
            uint_fast64_t getMecIndex(uint_fast64_t state);

            /*!
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
            std::pair<uint_fast64_t, uint_fast64_t> getNewIndex(uint_fast64_t state,
                                                                ValueType currentSumOfWeights,
                                                                uint_fast64_t currentWindowLength);
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
             * Get the maximum windows length.
             */
            uint_fast64_t getMaximumWindowsLength();

            /*!
             * Get a vector containing, for each MEC k, the meaning of each state in the new matrix corresponding to the
             * unfolding of the MEC k, expressed as a tuple (s, w, l) where s (state) is the state in the original matrix,
             * w (currentSumOfWeights) is the current sum of weights in the unfolding and l (currentWindowLength) is
             * the current window length in the unfolding.
             *
             * @param k the index of the MEC containing the state for which the meaning is explained.
             */
            std::vector<StateWeightWindowLength<ValueType>> getNewStatesMeaning(uint_fast64_t k);

            /*!
             * Prints the unfolding of the given mec to the given output stream.
             * @param out The output stream
             * @param k The index of the unfolded MEC
             */
            void printToStream(std::ostream& out, uint_fast64_t k);

        private:

            /*!
             * Maximum windows length
             */
            uint_fast64_t l_max;

            /*!
             * For each MEC in the MEC decomposition, this matrix represents the unfolded MEC for the window MP
             * objective.
             */
            std::vector<storm::storage::SparseMatrix<ValueType>> matrices;

            /*!
             * Vector containing the index of the MEC of each state.
             * Note that 0 is a special value indicating that the state does not belong to any MEC.
             */
            std::vector<uint_fast64_t> mecIndices;

            /*!
             * This vector contains the index of each original state s in the unfolding regarding to
             * the current window length l and the current sum of weights w in the unfolding of the MEC containing it.
             * usage: the index of the state s in the unfolding of the associated MEC is windowVector[s][l][w]
             */
            std::vector<std::vector<std::unordered_map<ValueType, uint_fast64_t>>> windowVector;

            /*!
             * For each MEC of the original MDP, this vector maps each new state to its pairs of
             * (successor states, probabilities) for each of its enabled actions that are ordered according to the
             * ordering of enabled actions of each state of the MEC.
             * Dimensions = (0: MEC, 1: state, 2: action, 3: pairs of (s', p) where p is the probability to go to s').
             */
            std::vector<std::vector<std::vector<std::vector<std::pair<uint_fast64_t, ValueType>>>>> newRowGroupEntries;

            /*!
             * Unfold a MEC from a given state. The result fields oldToNewStateMapping and newRowGroupEntries are
             * filled accordingly.
             *
             * @param state
             * @param currentSumOfWeights
             * @param l
             * @param originalMatrix
             * @param rewardModel
             * @param currentMec
             * @return the index of the input state in the new matrix of the current MEC
             */
            uint_fast64_t unfoldFrom(uint_fast64_t const& state, ValueType const& currentSumOfWeights, uint_fast64_t const& l,
                                     storm::storage::SparseMatrix<ValueType> const &originalMatrix,
                                     std::vector<ValueType> const& stateActionRewardsVector,
                                     storm::storage::MaximalEndComponent const &currentMec);
        };
    }
}

#endif //STORM_ECSUNFOLDING_H
