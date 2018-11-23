//
// Created by Florent Delgrange on 2018-11-23.
//

#include <storm/storage/SparseMatrix.h>
#include <storm/models/sparse/Model.h>
#include <iostream>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/MaximalEndComponentDecomposition.h>
#include "stochastic-windows/BndGoodWindowMP.h"

template <typename ValueType>
sw::BndGoodWindowMP::ECsUnfolding<ValueType>::ECsUnfolding(std::shared_ptr <storm::models::sparse::Mdp<ValueType>> mdp,
                                                std::string const &rewardModelName, uint_fast64_t const &l_max) {
    storm::storage::SparseMatrix<ValueType> originalMatrix = mdp->getTransitionMatrix();
    assert(mdp->hasRewardModel(rewardModelName));
    storm::models::sparse::StandardRewardModel<ValueType> rewardModel = mdp->getRewardModel(rewardModelName);
    assert(rewardModel.hasStateActionRewards());
    std::vector<ValueType> stateActionRewardsVector = rewardModel.getStateActionRewardVector();
    storm::storage::MaximalEndComponentDecomposition<ValueType> mecDecomposition(*mdp);

    matrices = std::vector<storm::storage::SparseMatrix<ValueType>>(mecDecomposition.size() + 1);
    // vector containing all data about the unfolding
    windowVector = std::vector<std::vector<std::unordered_map<ValueType, uint_fast64_t>>>(mdp->getNumberOfStates());
    storm::storage::MaximalEndComponent mec;
    // k = 0 is a special value meaning that the state does not belong to any MEC
    newRowGroupEntries.emplace_back();
    for (uint_fast64_t k = 1; k < mecDecomposition.size() + 1; ++k) {
        mec = mecDecomposition[k - 1];
        // initialize the new row group entries for the mec k
        newRowGroupEntries.emplace_back();
        storm::storage::SparseMatrixBuilder<ValueType> matrixBuilder;
        for (auto state : mec.getStateSet()) {
            mecIndices[state] = k;
            windowVector[state] = std::vector<std::unordered_map<ValueType, uint_fast64_t>>(l_max + 1);
        }
        // Unfold the kth MEC
        for (auto state: mec.getStateSet()) {
            unfoldFrom(state, 0., 0, l_max, originalMatrix, stateActionRewardsVector, mec);
        }
        // Build the new matrix w.r.t. the new row group entries computed during the unfolding
        uint_fast64_t newRow = 0;
        uint_fast64_t column;
        ValueType p;
        for (auto entries : newRowGroupEntries[k]) {
            matrixBuilder.newRowGroup(newRow);
            for (auto entry : entries) {
                std::tie(column, p) = entry;
                matrixBuilder.addNextValue(newRow, column, p);
                ++newRow;
            }
        }
        matrices[k] = matrixBuilder.build();
    }
}

template <typename ValueType>
uint_fast64_t sw::BndGoodWindowMP::ECsUnfolding<ValueType>::getMecIndex(uint_fast64_t state) {
    return mecIndices[state];
}

template <typename ValueType>
std::pair<uint_fast64_t, uint_fast64_t> sw::BndGoodWindowMP::ECsUnfolding<ValueType>::getNewIndex(uint_fast64_t state,
                                                                                                  ValueType currentSumOfWeights,
                                                                                                  uint_fast64_t currentWindowLength) {
    uint_fast64_t k = getMecIndex(state);
    if ( !k ) return std::make_pair(0, 0);
    auto keyValue = windowVector[state][currentWindowLength].find(currentSumOfWeights);
    if (keyValue == windowVector[state][currentWindowLength].end()) {
        return std::make_pair(k, 0);
    }
    else {
        return std::make_pair(k, keyValue->second);
    }
}

template <typename ValueType>
uint_fast64_t sw::BndGoodWindowMP::ECsUnfolding<ValueType>::unfoldFrom(u_int64_t const& state, ValueType const& currentSumOfWeights,
                                                                       uint_fast64_t const& l,
                                                                       uint_fast64_t const& l_max,
                                                                       storm::storage::SparseMatrix<ValueType> const &originalMatrix,
                                                                       std::vector<ValueType> const& stateActionRewardsVector,
                                                                       storm::storage::MaximalEndComponent const &currentMec){
    // k is the index of the current MEC containing the state
    uint_fast64_t k = getMecIndex(state);
    assert(newRowGroupEntries.size() > k);
    // Initialization of the new row group entries for the MEC k
    if (newRowGroupEntries[k].empty()) {
        // the index 0 is the index of the sink state corresponding to state (s, w, l) where l > l_max and w < 0
        newRowGroupEntries[k].emplace_back();
        newRowGroupEntries[k][0].push_back(std::make_pair(0, 1.));
    }
    // i is the row group index of (state, currentSomeOfWeights, l) in the new matrix
    uint_fast64_t i;
    std::tie(k, i) = getNewIndex(state, currentSumOfWeights, l);
    if ( !i ) {
        // If the state (state, currentSomOfWeights, l) does not yet exists, fill the newRowGroupEntries accordingly
        i = newRowGroupEntries[k].size();
        newRowGroupEntries[k].emplace_back();
        windowVector[state][l][currentSumOfWeights] = i;
        // assert( windowVector[state][l].find( currentSumOfWeights )->second == i );
        // if i was not in the map of weights, unfold the EC from the ith state s_i
        ValueType updatedSumOfWeights;
        for (auto action : currentMec.getChoicesForState(state)) {
            updatedSumOfWeights = currentSumOfWeights + stateActionRewardsVector[action];
            uint_fast64_t l_new = l + 1;
            // if reward(s_i, action) is >= 0 and l_new <= l_max, the window can be closed
            if (updatedSumOfWeights >= 0 and l_new <= l_max) {
                // the indices in the enumeration of enabled action for each state in the mec correspond to the
                // indices of rows in the original matrix
                for (auto entry : originalMatrix.getRow(action)) {
                    uint_fast64_t successorState = entry.getColumn();
                    ValueType p = entry.getValue();
                    // j is the index of successorState
                    uint_fast64_t j = unfoldFrom(successorState, 0., 0, l_max, originalMatrix, stateActionRewardsVector,
                                                 currentMec);
                    newRowGroupEntries[k][i].push_back(std::make_pair(j, p));
                }
            }
            else if (l_new <= l_max) {
                for (auto entry : originalMatrix.getRow(action)) {
                    uint_fast64_t successorState = entry.getColumn();
                    ValueType p = entry.getValue();
                    // j is the index of successorState
                    uint_fast64_t j = unfoldFrom(successorState, updatedSumOfWeights, l_new, l_max, originalMatrix,
                                                 stateActionRewardsVector, currentMec);
                    newRowGroupEntries[k][i].push_back(std::make_pair(j, p));
                }
            }
            else {
                // if s_i = (s, w, l_max) with w < 0, then the bounded window objective happens with
                // probability zero. Then, s_i transitions to the sink state.
                newRowGroupEntries[k][i].push_back(std::make_pair(0, 1.));
            }
        }
    }
    return i;
}

template <typename ValueType>
storm::storage::SparseMatrix<ValueType> sw::BndGoodWindowMP::ECsUnfolding<ValueType>::getUnfoldedMatrix(uint_fast64_t mec) {
    return matrices[mec - 1];
}
