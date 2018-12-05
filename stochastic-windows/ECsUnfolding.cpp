//
// Created by Florent Delgrange on 2018-11-23.
//

#include "stochastic-windows/ECsUnfolding.h"

template<typename ValueType>
sw::WindowMP::ECsUnfolding<ValueType>::ECsUnfolding(storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                                                    std::string const& rewardModelName,
                                                    uint_fast64_t const& l_max) {
    const storm::storage::SparseMatrix<ValueType> &originalMatrix = mdp.getTransitionMatrix();
    assert(mdp.hasRewardModel(rewardModelName));
    storm::models::sparse::StandardRewardModel<ValueType> rewardModel = mdp.getRewardModel(rewardModelName);
    assert(rewardModel.hasStateActionRewards());
    std::vector<ValueType> stateActionRewardsVector = rewardModel.getStateActionRewardVector();
    storm::storage::MaximalEndComponentDecomposition<ValueType> mecDecomposition(mdp);

    this->l_max = l_max;
    // Initialization of the matrices representing the unfolding of the ECS; note that the index 0 is a special one
    // (and is thus reserved) representing the non-ECs states
    matrices = std::vector<storm::storage::SparseMatrix<ValueType>>(mecDecomposition.size() + 1);
    // Initialization of the vector containing mec indices for each states
    mecIndices = std::vector<uint_fast64_t>(mdp.getNumberOfStates());
    // vector containing the data about states of the unfolding
    windowVector = std::vector<std::vector<std::unordered_map<ValueType, uint_fast64_t>>>(mdp.getNumberOfStates());
    storm::storage::MaximalEndComponent mec;
    // k = 0 is a special value meaning that the state does not belong to any MEC
    newRowGroupEntries.emplace_back();
    for (uint_fast64_t k = 1; k <= mecDecomposition.size(); ++ k) {
        mec = mecDecomposition[k - 1];
        // initialize the new row group entries for the kth mec
        newRowGroupEntries.emplace_back();
        storm::storage::SparseMatrixBuilder<ValueType> matrixBuilder(0, 0, 0, true, true);
        for (auto state : mec.getStateSet()) {
            mecIndices[state] = k;
            windowVector[state] = std::vector<std::unordered_map<ValueType, uint_fast64_t>>(l_max + 1);
        }
        // Unfold the kth MEC
        for (auto state: mec.getStateSet()) {
            unfoldFrom(state, 0., 0, originalMatrix, stateActionRewardsVector, mec);
        }
        // Build the new matrix w.r.t. the new row group entries computed during the unfolding
        uint_fast64_t newRow = 0;
        uint_fast64_t column;
        ValueType p;
        // iterate on states (new row groups) of the unfolding of the kth MEC
        for (const auto &newRowGroup : newRowGroupEntries[k]) {
            matrixBuilder.newRowGroup(newRow);
            // iterate on rows, i.e., enabled actions of the new state considered at the current iteration
            for (const auto &row : newRowGroup) {
                // iterate on each outgoing transition
                for (auto entry : row) {
                    std::tie(column, p) = entry;
                    matrixBuilder.addNextValue(newRow, column, p);
                }
                ++ newRow;
            }
        }
        matrices[k] = matrixBuilder.build();
    }
}

template <typename ValueType>
uint_fast64_t sw::WindowMP::ECsUnfolding<ValueType>::getMecIndex(uint_fast64_t state) {
    return mecIndices[state];
}

template <typename ValueType>
std::pair<uint_fast64_t, uint_fast64_t> sw::WindowMP::ECsUnfolding<ValueType>::getNewIndex(uint_fast64_t state,
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
uint_fast64_t sw::WindowMP::ECsUnfolding<ValueType>::unfoldFrom(uint_fast64_t const& state,
                                                                ValueType const& currentSumOfWeights,
                                                                uint_fast64_t const& l,
                                                                storm::storage::SparseMatrix<ValueType> const &originalMatrix,
                                                                std::vector<ValueType> const& stateActionRewardsVector,
                                                                storm::storage::MaximalEndComponent const &currentMec){
    // k is the index of the current MEC containing the state
    uint_fast64_t k = getMecIndex(state);
    assert(newRowGroupEntries.size() > k);
    // Initialization of the new row group entries for the MEC k
    if (newRowGroupEntries[k].empty()) {
        // the index 0 is the index of the sink state corresponding to states (s, w, l) where l > l_max and w < 0
        newRowGroupEntries[k].emplace_back();
        newRowGroupEntries[k][0].push_back({std::make_pair(0, 1.)});
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
        // as i was not in the map of weights, unfold the EC from the ith state s_i
        ValueType updatedSumOfWeights;
        for (auto action : currentMec.getChoicesForState(state)) {
            updatedSumOfWeights = currentSumOfWeights + stateActionRewardsVector[action];
            uint_fast64_t l_new = l + 1;
            // add the current action to the new row group entries
            newRowGroupEntries[k][i].emplace_back();
            uint_fast64_t newAction = newRowGroupEntries[k][i].size() - 1;
            // if reward(s_i, action) is >= 0 and l_new <= l_max, the window can be closed
            if (updatedSumOfWeights >= 0 and l_new <= l_max) {
                // the indices in the enumeration of enabled action for each state in the mec correspond to the
                // indices of rows in the original matrix
                for (const auto &entry : originalMatrix.getRow(action)) {
                    uint_fast64_t successorState = entry.getColumn();
                    ValueType p = entry.getValue();
                    // j is the index of successorState
                    uint_fast64_t j = unfoldFrom(successorState, 0., 0, originalMatrix, stateActionRewardsVector,
                                                 currentMec);
                    newRowGroupEntries[k][i][newAction].push_back(std::make_pair(j, p));
                }
            }
            else if (l_new <= l_max) {
                for (const auto &entry : originalMatrix.getRow(action)) {
                    uint_fast64_t successorState = entry.getColumn();
                    ValueType p = entry.getValue();
                    // j is the index of successorState
                    uint_fast64_t j = unfoldFrom(successorState, updatedSumOfWeights, l_new, originalMatrix,
                                                 stateActionRewardsVector, currentMec);
                    newRowGroupEntries[k][i][newAction].push_back(std::make_pair(j, p));
                }
            }
            else {
                // if s_i = (s, w, l_max) with w < 0, then the bounded window objective happens with
                // probability zero. Then, s_i transitions to the sink state.
                newRowGroupEntries[k][i][newAction] = {std::make_pair(0, 1.)};
                // Since the bound l_max is exceeded, we can ignore other useless actions necessarily leading to the
                // sink state.
                break;
            }
        }
    }
    return i;
}

template <typename ValueType>
storm::storage::SparseMatrix<ValueType>& sw::WindowMP::ECsUnfolding<ValueType>::getUnfoldedMatrix(uint_fast64_t mec) {
    return this->matrices[mec];
}

template<typename ValueType>
uint_fast64_t sw::WindowMP::ECsUnfolding<ValueType>::getNumberOfUnfoldedECs() {
    return this->matrices.size() - 1;
}

template<typename ValueType>
uint_fast64_t sw::WindowMP::ECsUnfolding<ValueType>::getMaximumWindowsLength() {
    return l_max;
}

template<typename ValueType>
std::vector<sw::WindowMP::StateWeightWindowLength<ValueType>>
sw::WindowMP::ECsUnfolding<ValueType>::getNewStatesMeaning(uint_fast64_t k) {

    std::vector<sw::WindowMP::StateWeightWindowLength<ValueType>> unfoldedStates(newRowGroupEntries[k].size());

    for (uint_fast64_t state = 0; state < mecIndices.size(); ++ state) {
        if (mecIndices[state] == k) {
            for (uint_fast64_t l = 0; l <= l_max; ++ l) {
                for (const auto &keyValue : windowVector[state][l]) {
                    unfoldedStates[keyValue.second].state = state;
                    unfoldedStates[keyValue.second].currentSumOfWeights = keyValue.first;
                    unfoldedStates[keyValue.second].currentWindowLength = l;
                }
            }
        }
    }

    return unfoldedStates;
}

template<typename ValueType>
void sw::WindowMP::ECsUnfolding<ValueType>::printToStream(std::ostream &out, uint_fast64_t k) {
    out << this->getUnfoldedMatrix(k) << "\n";
    out << "where" << "\n";
    std::vector<sw::WindowMP::StateWeightWindowLength<ValueType>>
            newStatesMeaning = this->getNewStatesMeaning(k);
    out << "state 0 is ⊥" << "\n";
    for(uint_fast64_t state = 1; state < this->getUnfoldedMatrix(k).getRowGroupCount(); ++ state) {
        std::ostringstream stream;
        stream << "(s" << newStatesMeaning[state].state << ", " <<
               newStatesMeaning[state].currentSumOfWeights << ", " <<
               newStatesMeaning[state].currentWindowLength << ")";
        out << "state " << state << " is " << stream.str() << "\n";
    }
}

template class sw::WindowMP::ECsUnfolding<double>;
template class sw::WindowMP::ECsUnfolding<storm::RationalNumber>;
