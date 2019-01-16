//
// Created by Florent Delgrange on 2019-01-14.
//

#include "UnfoldingMeanPayoff.h"

template<typename ValueType>
sw::DirectFixedWindow::UnfoldingMeanPayoff<ValueType>::UnfoldingMeanPayoff(
        storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>>& mdp,
        std::string const& rewardModelName,
        uint_fast64_t const& l_max,
        storm::storage::BitVector const& initialStates,
        std::vector<std::vector<uint_fast64_t>> enabledActions)
        : originalMatrix(mdp.getTransitionMatrix()),
          enabledActions(enabledActions) {

    storm::models::sparse::StandardRewardModel<ValueType> rewardModel = mdp.getRewardModel(rewardModelName);

    assert(initialStates.size() == mdp.getNumberOfStates());
    assert(enabledActions.size() == mdp.getNumberOfStates());
    assert(mdp.hasRewardModel(rewardModelName));
    assert(rewardModel.hasStateActionRewards());

    this->l_max = l_max;
    this->stateActionRewardsVector = rewardModel.getStateActionRewardVector();
    // vector containing data about states of the unfolding
    this->windowVector = std::vector<std::vector<std::unordered_map<ValueType, uint_fast64_t>>>(mdp.getNumberOfStates());
    newRowGroupEntries.emplace_back();
    for (uint_fast64_t state = 0; state < mdp.getNumberOfStates(); ++state) {
        this->windowVector[state] = std::vector<std::unordered_map<ValueType, uint_fast64_t>>(l_max);
    }
    // Unfold the MDP from the initial states
    for (auto state: initialStates) {
        unfoldFrom(state, 0., 0);
    }
    // this line allows to have the same number of columns and rowGroups in the sparse matrix
    newRowGroupEntries[0][0].push_back(std::make_pair(newRowGroupEntries.size() - 1, storm::utility::zero<ValueType>()));
    // Build the new matrix w.r.t. the new row group entries computed during the unfolding
    uint_fast64_t newRow = 0;
    uint_fast64_t column;
    ValueType p;
    storm::storage::SparseMatrixBuilder<ValueType> matrixBuilder(0, 0, 0, true, true);
    // iterate on states (new row groups) of the unfolding
    for (const auto &newRowGroup : newRowGroupEntries) {
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
        this->matrix = matrixBuilder.build();
        this->matrix.makeRowDirac(0, 0);
        assert(this->matrix.getRowGroupCount() == this->matrix.getColumnCount());
    }
}

template<typename ValueType>
sw::DirectFixedWindow::UnfoldingMeanPayoff<ValueType>::UnfoldingMeanPayoff(
        storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>>& mdp,
        std::string const& rewardModelName,
        uint_fast64_t const& l_max,
        storm::storage::BitVector const& initialStates)
        : sw::DirectFixedWindow::UnfoldingMeanPayoff<ValueType>::UnfoldingMeanPayoff(
                mdp, rewardModelName, l_max, initialStates,
                sw::DirectFixedWindow::UnfoldingMeanPayoff<ValueType>::enableAllActions(mdp.getTransitionMatrix())) {}

template<typename ValueType>
std::vector<std::vector<uint_fast64_t>> const sw::DirectFixedWindow::UnfoldingMeanPayoff<ValueType>::enableAllActions(
        storm::storage::SparseMatrix<ValueType> const& originalMatrix) {

    std::vector<std::vector<uint_fast64_t>> enabledActions(originalMatrix.getRowGroupCount());
    for (uint_fast64_t state = 0; state < originalMatrix.getRowGroupCount(); ++ state) {
        enabledActions[state] = std::vector<uint_fast64_t>(originalMatrix.getRowGroup(state).getNumberOfEntries());
        for (uint_fast64_t action = 0; action < enabledActions[state].size(); ++ action) {
            enabledActions[state][action] = action;
        }
    }
    return enabledActions;
}

template <typename ValueType>
uint_fast64_t sw::DirectFixedWindow::UnfoldingMeanPayoff<ValueType>::getNewIndex(
        uint_fast64_t state,
        ValueType currentSumOfWeights,
        uint_fast64_t currentWindowSize) {

    auto keyValue = this->windowVector[state][currentWindowSize].find(currentSumOfWeights);
    if (keyValue == this->windowVector[state][currentWindowSize].end()) {
        // note that 0 is a special value (it represents the sink state in the unfolding)
        return 0;
    }
    else {
        return keyValue->second;
    }
}

template <typename ValueType>
uint_fast64_t sw::DirectFixedWindow::UnfoldingMeanPayoff<ValueType>::unfoldFrom(
        uint_fast64_t const& state,
        ValueType const& currentSumOfWeights,
        uint_fast64_t const& l) {

    // Initialization
    if (this->newRowGroupEntries.empty()) {
        // the index 0 is the index of the sink state corresponding to states (s, w, l) where l >= l_max and w < 0
        this->newRowGroupEntries.emplace_back();
        this->newRowGroupEntries[0].push_back({std::make_pair(0, 1.)});
    }
    // i is the row group index of (state, currentSumOfWeights, l) in the new matrix
    uint_fast64_t i = this->getNewIndex(state, currentSumOfWeights, l);
    if ( !i ) {
        // If the unfolding state (state, currentSumOfWeights, l) does not yet exist, fill newRowGroupEntries accordingly
        i = newRowGroupEntries.size();
        newRowGroupEntries.emplace_back();
        windowVector[state][l][currentSumOfWeights] = i;
        // as i was not in the map of weights, unfold the EC from the ith state s_i
        ValueType updatedSumOfWeights;
        uint_fast64_t l_new = l + 1;
        for (auto action : this->enabledActions[state]) {
            updatedSumOfWeights = currentSumOfWeights + stateActionRewardsVector[action];
            // add the current action to the new row group entries
            this->newRowGroupEntries[i].emplace_back();
            uint_fast64_t newAction = this->newRowGroupEntries[i].size() - 1;
            // if reward(s_i, action) is >= 0 and l_new <= l_max, the window is closed
            if (updatedSumOfWeights >= 0 and l_new <= l_max) {
                // the indices in the enumeration of enabled action for each state correspond to the
                // indices of rows in the original matrix
                for (const auto &entry : this->matrix.getRow(action)) {
                    uint_fast64_t successorState = entry.getColumn();
                    ValueType p = entry.getValue();
                    // j is the index of successorState
                    uint_fast64_t j = unfoldFrom(successorState, 0., 0);
                    this->newRowGroupEntries[i][newAction].push_back(std::make_pair(j, p));
                }
            }
            else if (l_new < l_max) {
                for (const auto &entry : this->originalMatrix.getRow(action)) {
                    uint_fast64_t successorState = entry.getColumn();
                    ValueType p = entry.getValue();
                    // j is the index of successorState
                    uint_fast64_t j = unfoldFrom(successorState, updatedSumOfWeights, l_new);
                    this->newRowGroupEntries[i][newAction].push_back(std::make_pair(j, p));
                }
            }
            else {
                // if s_i = (s, w, l_max) with w < 0, then the bounded window objective happens with
                // probability zero. Then, s_i is the actual sink state.
                this->newRowGroupEntries[i][newAction] = {std::make_pair(0, 1.)};
            }
        }
    }
    return i;
}

template<typename ValueType>
storm::storage::SparseMatrix<ValueType>& sw::DirectFixedWindow::UnfoldingMeanPayoff<ValueType>::getMatrix() {
    return this->matrix;
}

template<typename ValueType>
std::vector<sw::DirectFixedWindow::StateWeightWindowSize<ValueType>>
sw::DirectFixedWindow::UnfoldingMeanPayoff<ValueType>::getNewStatesMeaning() {

    std::vector<sw::DirectFixedWindow::StateWeightWindowSize<ValueType>> unfoldingStates(this->newRowGroupEntries.size());

    for (uint_fast64_t state = 0; state < this->originalMatrix.getRowGroupCount(); ++ state) {
        for (uint_fast64_t l = 0; l < l_max; ++ l) {
            for (const auto &keyValue : this->windowVector[state][l]) {
                unfoldingStates[keyValue.second].state = state;
                unfoldingStates[keyValue.second].currentSumOfWeights = keyValue.first;
                unfoldingStates[keyValue.second].currentWindowSize = l;
            }
        }
    }
    return unfoldingStates;
}

template class sw::DirectFixedWindow::UnfoldingMeanPayoff<double>;
template class sw::DirectFixedWindow::UnfoldingMeanPayoff<storm::RationalNumber>;
