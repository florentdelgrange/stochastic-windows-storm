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

        struct OldToNewStateMapping {
            uint_fast64_t mecIndex;
            std::vector<std::unordered_map<ValueType, uint_fast64_t>>> windowVector;
        };

        struct ECsUnfolding {
            // For each MEC in the MEC decomposition, create a matrix representing the unfolded MEC for the window MP
            // objective.
            std::vector<storm::storage::SparseMatrix<ValueType>> matrices;
            // For each MEC of the original MDP, this vector maps each new state to its pairs of
            // (successor states, probabilities) for each of its enabled actions that are ordered according to the
            // ordering of enabled actions of each state of the MEC.
            std::vector<std::vector<std::vector<std::pair<uint_fast64_t, ValueType>>>> newRowGroupEntries;
            // Array mapping for each state of the original MDP a pair containing, at first position, the index of
            // the new matrix containing it (i.e., the index of the corresponding MEC), and, at second position,
            // an array of size l_max containing in each position l a hash table mapping a (current negative) weight w
            // to an index in the new matrix corresponding to the the following pair :
            // (s, w, l), where s is the original state of the MDP, w is the current value of the window (sum of
            // weights), and l is the current number of steps of the window.
            std::vector<std::pair<uint_fast64_t, std::vector<std::unordered_map<ValueType, uint_fast64_t>>>> oldToNewStateMapping;
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
                uint_fast64_t const &l_max) {
            storm::storage::SparseMatrix<ValueType> originalMatrix = mdp->getTransitionMatrix();
            assert(mdp->hasRewardModel(rewardModelName));
            storm::models::sparse::StandardRewardModel<ValueType> rewardModel = mdp->getRewardModel(rewardModelName);
            assert(rewardModel.hasStateActionRewards());
            storm::storage::MaximalEndComponentDecomposition<ValueType> mecDecomposition(*mdp);

            // Initialization phase
            // Array containing the sparse matrix for each MEC.
            ECsUnfolding result;
            result.matrices = std::vector<storm::storage::SparseMatrix<ValueType>>(mecDecomposition.size() + 1);
            // Matrix containing all data about the unfolding
            result.oldToNewStateMapping = std::vector<std::pair<uint_fast64_t, std::vector<std::unordered_map<ValueType, uint_fast64_t>>>>(mdp->getNumberOfStates());
            storm::storage::MaximalEndComponent mec;
            // k = 0 is a special value meaning that the state does not belong to any MEC
            result.newRowGroupEntries.emplace_back();
            for (uint_fast64_t k = 1; k <= mecDecomposition.size(); ++k) {
                mec = mecDecomposition[k - 1];
                // initialize the new row group entries for the mec k
                result.newRowGroupEntries.emplace_back();
                storm::storage::SparseMatrixBuilder<ValueType> matrixBuilder;
                for (auto state : mec.getStateSet()) {
                    // Initialize a pair with the mec containing state and an empty vector of size l_max
                    result.oldToNewStateMapping[state] = std::make_pair(k, std::vector<std::unordered_map<ValueType, uint_fast64_t>>(l_max + 1));
                }
                // Unfold the kth MEC
                for (auto state: mec.getStateSet()) {
                    unfoldFrom(state, 0., 0, l_max, originalMatrix, rewardModel, mec, result);
                }
                // Build the new matrix w.r.t. the new row group entries computed during the unfolding
                uint_fast64_t newRow = 0;
                uint_fast64_t column;
                ValueType p;
                for (auto newRowGroupEntries : result.newRowGroupEntries[k]) {
                    matrixBuilder.newRowGroup(newRow);
                    for (auto entry : newRowGroupEntries) {
                        std::tie(column, p) = entry;
                        matrixBuilder.addNextValue(newRow, column, p);
                        ++newRow;
                    }
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
        static uint_fast64_t unfoldFrom(u_int64_t const& state, ValueType const& currentSumOfWeights, uint_fast64_t const& l,
                    uint_fast64_t const& l_max,
                    storm::storage::SparseMatrix<ValueType> const &originalMatrix,
                    storm::models::sparse::StandardRewardModel<ValueType> const &rewardModel,
                    storm::storage::MaximalEndComponent const &currentMec,
                    ECsUnfolding result){
            std::cout << "unfolding: (" << state << ", " << currentSumOfWeights << ", " << l << ")" << endl;
            // k is the index of the current MEC containing the state
            uint_fast64_t k = result.oldToNewStateMapping[state].first;
            // window vector of size l_max
            std::vector<std::unordered_map<ValueType, uint_fast64_t>> windowVector = result.oldToNewStateMapping[state].second;
            assert(result.newRowGroupEntries.size() >= k + 1);
            // Initialization of the new row group entries for the MEC k
            if (result.newRowGroupEntries[k].empty()) {
                // the index 0 is the index of the sink state corresponding to state (s, w, l) where l > l_max and w < 0
                result.newRowGroupEntries[k].emplace_back();
                result.newRowGroupEntries[k][0].push_back(std::make_pair(0, 1.));
            }
            //std::cout << "    " << currentSumOfWeights << " in weightsMap? " << (weightsMap.count(currentSumOfWeights) > 0) << endl;
            std::cout << "    current content of weightsMap(" << result.oldToNewStateMapping[state].second[l].size() << ") : ";
            for (auto keyValue : result.oldToNewStateMapping[state].second[l]){
                std::cout << "(" << keyValue.first << ", " << keyValue.second << ")";
            }
            std::cout << std::endl;
            // i is the row group index of (state, currentSomeOfWeights, l) in the new matrix
            uint_fast64_t i;
            auto keyValue = result.oldToNewStateMapping[state].second[l].find(currentSumOfWeights);
            if (keyValue == result.oldToNewStateMapping[state].second[l].end()) {
                // If the state (state, currentSomOfWeights, l) does not yet exists, fill the newRowGroupEntries accordingly
                i = result.newRowGroupEntries[k].size();
                result.newRowGroupEntries[k].emplace_back();
                result.oldToNewStateMapping[state].second[l][currentSumOfWeights] = i;
                assert( result.oldToNewStateMapping[state].second[l].find( currentSumOfWeights )->second == i );
                // if i was not in the map of weights, unfold the EC from the ith state s_i
                std::cout << "    updated content of weightsMap(" << result.oldToNewStateMapping[state].second[l].size() << ") : ";
                for (auto keyValue : result.oldToNewStateMapping[state].second[l]){
                    std::cout << "(" << keyValue.first << ", " << keyValue.second << ")";
                }
                std::cout << endl;
                ValueType updatedSumOfWeights;
                std::cout << "    weightsMap size=" << result.oldToNewStateMapping[state].second[l].size() << endl;
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
                            result.newRowGroupEntries[k][i].push_back(std::make_pair(j, p));
                        }
                    }
                    else if (l_new <= l_max) {
                        for (auto entry : originalMatrix.getRow(action)) {
                            uint_fast64_t successorState = entry.getColumn();
                            ValueType p = entry.getValue();
                            // j is the index of successorState
                            uint_fast64_t j = unfoldFrom(successorState, updatedSumOfWeights, l_new, l_max, originalMatrix, rewardModel, currentMec, result);
                            result.newRowGroupEntries[k][i].push_back(std::make_pair(j, p));
                        }
                    }
                    else {
                        // if s_i = (s, w, l_max) with w < 0, then the bounded window objective happens with
                        // probability zero. Then, s_i transitions to the sink state.
                        result.newRowGroupEntries[k][i].push_back(std::make_pair(0, 1.));
                    }
                }
            }
            else {
                i = keyValue->second;
            }
            return i;
        }
    };

}