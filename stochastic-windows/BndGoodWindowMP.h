//
// Created by Florent Delgrange on 09/11/2018.
//

#include <storm/storage/SparseMatrix.h>
#include <storm/models/sparse/Model.h>
#include <iostream>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/MaximalEndComponentDecomposition.h>

#ifndef STORM_BNDGOODWINDOWMP_H
#define STORM_BNDGOODWINDOWMP_H

#endif //STORM_BNDGOODWINDOWMP_H

namespace sw {

    template <typename ValueType>

    class BndGoodWindowMP {
    public:

        struct ECsUnfolding {
            // For each MEC in the MEC decomposition, create a matrix representing the unfolded MEC for the window MP
            // objective.
            std::vector<storm::storage::SparseMatrix<ValueType>> matrices;
            // For each MEC of the original MDP, this vector maps each new state to its pairs of
            // (successor states, probabilities) for each of its enabled actions that are ordered according to the
            // ordering of enabled actions of each state of the MEC.
            std::vector<std::vector<std::vector<std::tuple<uint_fast64_t, ValueType>>>> newRowGroupEntries;
            // Array mapping for each state of the original MDP a tuple containing, at the first position, the index of
            // the new matrix containing it (i.e., the index of the corresponding MEC), and, at the second position,
            // an array of size l_max containing in each position l a hash table mapping a (current negative) weight w
            // to an index in the new matrix corresponding to the the following tuple :
            // (s, w, l), where s is the original state of the MDP, w is the current value of the window (sum of
            // weights), and l is the current number of steps of the window.
            std::vector<std::tuple<uint_fast64_t, std::vector<std::map<ValueType, uint_fast64_t>>>> oldToNewStateMapping;
        };

        /**
         * Unfold the end components of the mdp entered in parameter for the window mean payoff problem w.r.t. the
         * maximal windows' length l_max.
         *
         * @param mdp Markov decision process for which the ECs will be unfolded
         * @param rewardModelName name of the reward model following which the mdp will be unfolded
         * @param l_max the maximum length of windows to consider
         * @return
         */
        static ECsUnfolding unfoldECs(std::shared_ptr<storm::models::sparse::Mdp<ValueType>> mdp,
                std::string const& rewardModelName,
                uint_fast64_t l_max) {
            storm::storage::SparseMatrix<ValueType> originalMatrix = mdp->getTransitionMatrix();
            assert(mdp->hasRewardModel(rewardModelName));
            storm::models::sparse::StandardRewardModel<ValueType> rewardModel = mdp->getRewardModel(rewardModelName);
            assert(rewardModel.hasStateActionRewards());
            storm::storage::MaximalEndComponentDecomposition<ValueType> mecDecomposition(*mdp);

            // Initialization phase
            // Array containing the sparse matrix for each MEC.
            ECsUnfolding result;
            result.matrices = std::vector<storm::storage::SparseMatrix<ValueType>>(mecDecomposition.size());
            // Matrix containing all data about the unfolding
            result.oldToNewStateMapping = std::vector<std::tuple<uint_fast64_t, std::vector<std::map<ValueType, uint_fast64_t>>>>(mdp->getNumberOfStates());
            storm::storage::MaximalEndComponent mec;
            for (int k = 0; k < mecDecomposition.size(); ++k){
                 mec = mecDecomposition[k];
                 // initialize the new row group entries for the mec k
                 result.newRowGroupEntries.emplace_back();
                 storm::storage::SparseMatrixBuilder<ValueType> matrixBuilder = storm::storage::SparseMatrixBuilder<ValueType>::SparseMatrixBuilder();
                 for (auto state : mec.getStateSet()) {
                     // Initialize a tuple with the mec containing state and an empty vector of size l_max
                     result.oldToNewStateMapping[state] = std::make_tuple(k, std::vector<std::map<ValueType, uint_fast64_t>>(l_max + 1));
                 }
                 // Unfold the kth MEC
                 for (auto state: mec.getStateSet()) {
                     unfoldFrom(state, 0., 0, l_max, originalMatrix, rewardModel, mec, result);
                 }
                result.matrices[k] = matrixBuilder.build();
            }
            return result;
        };

    private:

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
         * @param result
         * @return the index of the input state in the new matrix of the current MEC
         */
        static uint_fast64_t unfoldFrom(u_int64_t state, ValueType const &currentSumOfWeights, uint_fast64_t l,
                    uint_fast64_t l_max,
                    storm::storage::SparseMatrix<ValueType> const &originalMatrix,
                    storm::models::sparse::StandardRewardModel<ValueType> const &rewardModel,
                    storm::storage::MaximalEndComponent const &currentMec,
                    ECsUnfolding const &result){
            // k is the index of the current MEC containing the state
            uint_fast64_t k;
            // window vector of size l_max
            std::vector<std::map<ValueType, uint_fast64_t>> windowVector;
            std::tie(k, windowVector) = result.oldToNewStateMapping[state];
            assert(result.newRowGroupEntries.size() >= k);
            // Initialization of the new row group entries for the MEC k
            if (result.newRowGroupEntries[k].empty()) {
                // the index 0 is the index of the sink state corresponding to state (s, w, l) where l > l_max and w < 0
                result.newRowGroupEntries[k].push_back(std::make_tuple(0, 1.));
            }
            std::map<ValueType, uint_fast64_t> weightsMap = windowVector[l];
            // i is the row group index of (state, currentSomeOfWeights, l) in the new matrix
            auto i = weightsMap.find(currentSumOfWeights);
            if (i == weightsMap.end()) {
                // If the state (state, currentSomOfWeights, l) does not yet exists, fill the newRowGroupEntries accordingly
                i = result.newRowGroupEntries[k].size();
                result.newRowGroupEntries[k].emplace_back();
                weightsMap[currentSumOfWeights] = i;
                // if i was not in the map of weights, unfold the EC from the ith state s_i
                ValueType updatedSumOfWeights;
                for (auto action : currentMec.getChoicesForState(state)) {
                    updatedSumOfWeights = currentSumOfWeights + rewardModel.getStateActionRewardVector()[action];
                    uint_fast64_t l_new = l + 1;
                    // if reward(s_i, action) is >= 0 and l_new <= l_max, the window can be closed
                    if (updatedSumOfWeights >= 0 and l_new <= l_max) {
                        // the indices in the enumeration of enabled action for each state in the mec correspond to the
                        // indices of rows in the original matrix
                        for (auto entry : originalMatrix.getRow(action)) {
                            uint_fast64_t successorState = entry.getColumn();
                            ValueType p = entry.getValue();
                            // j is the index of successorState
                            uint_fast64_t j = unfoldFrom(successorState, 0., 0, l_max, originalMatrix, rewardModel, currentMec, result);
                            result.newRowGroupEntries[k][i].push_back(std::make_tuple(j, p));
                        }
                    }
                    else if (l_new <= l_max) {
                        for (auto entry : originalMatrix.getRow(action)) {
                            uint_fast64_t successorState = entry.getColumn();
                            ValueType p = entry.getValue();
                            // j is the index of successorState
                            uint_fast64_t j = unfoldFrom(successorState, updatedSumOfWeights, l, l_max, originalMatrix, rewardModel, currentMec, result);
                            result.newRowGroupEntries[k][i].push_back(std::make_tuple(j, p));
                        }
                    }
                    else {
                        // if s_i = (s, w, l_max) with w < 0, then the bounded window objective happens with
                        // probability zero. Then, s_i transitions to the sink state.
                        result.newRowGroupEntries[k][i].push_back(std::make_tuple(0, 1.));
                    }
                }
            }
            return i;
        }
    };

}