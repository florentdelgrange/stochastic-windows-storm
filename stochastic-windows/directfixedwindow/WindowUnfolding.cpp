#include <utility>

//
// Created by Florent Delgrange on 2019-01-14.
//

#include "WindowUnfolding.h"

namespace sw {
    namespace DirectFixedWindow {

        template<typename ValueType>
        WindowUnfolding<ValueType>::WindowUnfolding(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const& rewardModelName,
                uint_fast64_t const &l_max,
                storm::storage::BitVector const& enabledActions)
                : l_max(l_max),
                  mdp(mdp),
                  originalMatrix(mdp.getTransitionMatrix()),
                  enabledActions(enabledActions),
                  rewardModel(mdp.getRewardModel(rewardModelName)) {

            STORM_LOG_ASSERT(this->l_max > 0, "The maximum window size must be greater than 0");
            STORM_LOG_ASSERT(this->enabledActions.size() == mdp.getNumberOfChoices(),
                    "The size of the BitVector representing the set of enabled actions must be the number of actions in the system");

            // vector containing data about states of the unfolding
            this->windowVector = std::vector<std::vector<std::unordered_map<ValueType, uint_fast64_t>>>(mdp.getNumberOfStates());
            for (uint_fast64_t state = 0; state < mdp.getNumberOfStates(); ++state) {
                this->windowVector[state] = std::vector<std::unordered_map<ValueType, uint_fast64_t>>(l_max);
            }
        }

        template<typename ValueType>
        WindowUnfolding<ValueType>::WindowUnfolding(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const& rewardModelName,
                uint_fast64_t const &l_max)
                : WindowUnfolding(mdp, rewardModelName, l_max,
                        storm::storage::BitVector(mdp.getTransitionMatrix().getRowCount(), true)) {}

        template<typename ValueType>
        WindowUnfoldingMeanPayoff<ValueType>::WindowUnfoldingMeanPayoff(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const &rewardModelName, uint_fast64_t const &l_max,
                storm::storage::BitVector const &initialStates,
                storm::storage::BitVector const& enabledActions)
                : WindowUnfolding<ValueType>(mdp, rewardModelName, l_max, enabledActions) {
            assert(initialStates.size() == mdp.getNumberOfStates());
            WindowUnfolding<ValueType>::generateMatrix(initialStates);
        }

        template<typename ValueType>
        WindowUnfoldingMeanPayoff<ValueType>::WindowUnfoldingMeanPayoff(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const &rewardModelName, uint_fast64_t const &l_max,
                storm::storage::BitVector const &initialStates)
                : WindowUnfoldingMeanPayoff<ValueType>(mdp, rewardModelName, l_max, initialStates,
                        storm::storage::BitVector(mdp.getTransitionMatrix().getRowCount(), true)) {}

        template<typename ValueType>
        WindowUnfoldingParity<ValueType>::WindowUnfoldingParity(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const &rewardModelName, uint_fast64_t const &l_max,
                storm::storage::BitVector const &initialStates,
                storm::storage::BitVector const& enabledActions)
                : WindowUnfolding<ValueType>(mdp, rewardModelName, l_max, enabledActions) {
            assert(initialStates.size() == mdp.getNumberOfStates());
            WindowUnfolding<ValueType>::generateMatrix(initialStates);
        }

        template<typename ValueType>
        WindowUnfoldingParity<ValueType>::WindowUnfoldingParity(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const &rewardModelName, uint_fast64_t const &l_max,
                storm::storage::BitVector const &initialStates)
                : WindowUnfoldingParity<ValueType>(mdp, rewardModelName, l_max, initialStates,
                        storm::storage::BitVector(mdp.getTransitionMatrix().getRowCount(), true)) {}

        template<typename ValueType>
        void WindowUnfolding<ValueType>::generateMatrix(storm::storage::BitVector const &initialStates) {
            // Unfold the MDP from the initial states
            for (auto state: initialStates) {
                unfoldFrom(state, initialStateValue(state), 0);
            }
            // this line allows to have the same number of columns and rowGroups in the sparse matrix
            this->newRowGroupEntries[0][0].push_back(
                    std::make_pair(newRowGroupEntries.size() - 1, storm::utility::zero<ValueType>()));
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
                    ++newRow;
                }
            }
            this->matrix = matrixBuilder.build();
            this->matrix.makeRowDirac(0, 0);
            assert(this->matrix.getRowGroupCount() == this->matrix.getColumnCount());
        }

        template<typename ValueType>
        ValueType WindowUnfoldingMeanPayoff<ValueType>::initialStateValue(uint_fast64_t initialState) {
            return storm::utility::zero<ValueType>();
        }

        template<typename ValueType>
        ValueType WindowUnfoldingParity<ValueType>::initialStateValue(uint_fast64_t initialState) {
            return this->rewardModel.getStateReward(initialState);
        }

        template<typename ValueType>
        uint_fast64_t WindowUnfolding<ValueType>::getNewIndex(
                uint_fast64_t state,
                ValueType value,
                uint_fast64_t currentWindowSize) {

            auto keyValue = this->windowVector[state][currentWindowSize].find(value);
            if (keyValue == this->windowVector[state][currentWindowSize].end()) {
                // 0 is a special value (it represents the sink state in the unfolding);
                // it means that the input (state, value, currentWindowSize) does not yet exist in the unfolding.
                return 0;
            } else {
                return keyValue->second;
            }
        }

        template<typename ValueType>
        uint_fast64_t WindowUnfolding<ValueType>::getMaxNumberOfMemoryStatesRequired() {
            // current max number of memory states required for a memory structure
            uint_fast64_t M = 0;
            for (uint_fast64_t s = 0; s < this->originalMatrix.getRowGroupCount(); ++ s) {
                // current number of memory states for s
                uint_fast64_t memory = 0;
                for (uint_fast64_t l = 0; l < this->l_max; ++ l) {
                    memory += this->windowVector[s][l].size();
                }
                if (M < memory) {
                    M = memory;
                }
            }
            return M + 1; // the additional memory state is linked to the sink state in the unfolding
        }

        template <typename ValueType>
        WindowMemory<ValueType> WindowUnfolding<ValueType>::generateMemory() {
            uint_fast64_t M = this->getMaxNumberOfMemoryStatesRequired();
            storm::storage::MemoryStructureBuilder<ValueType> memoryBuilder(M, this->mdp);
            // initialize the mapping
            std::vector<std::vector<std::unordered_map<ValueType, uint_fast64_t>>>
            unfoldingToMemoryStatesMapping(this->mdp.getNumberOfStates());
            for (uint_fast64_t state = 0; state < mdp.getNumberOfStates(); ++ state) {
                unfoldingToMemoryStatesMapping[state] = std::vector<std::unordered_map<ValueType, uint_fast64_t>>(l_max);
            }

            std::vector<uint_fast64_t> currentMemoryIndices(M, 0);
            for (uint_fast64_t unfoldingState = 1; unfoldingState < this->matrix.getRowGroupCount(); ++ unfoldingState) {
                uint_fast64_t s = this->matrix.getRowGroupCount(), l = l_max;
                ValueType x;
            }
        }

        template<typename ValueType>
        uint_fast64_t WindowUnfoldingMeanPayoff<ValueType>::getInitialState(uint_fast64_t originalInitialState) {
            return this->getNewIndex(originalInitialState, this->initialStateValue(originalInitialState), 0);
        }

        template<typename ValueType>
        uint_fast64_t WindowUnfoldingParity<ValueType>::getInitialState(uint_fast64_t originalInitialState) {
            return this->getNewIndex(originalInitialState, this->initialStateValue(originalInitialState), 0);
        }

        template<typename ValueType>
        storm::storage::SparseMatrix<ValueType> &WindowUnfolding<ValueType>::getMatrix() {
            return this->matrix;
        }

        template<typename ValueType>
        std::vector<StateValueWindowSize<ValueType>>
        WindowUnfolding<ValueType>::getNewStatesMeaning() {

            std::vector<StateValueWindowSize<ValueType>> unfoldingStates(
                    this->newRowGroupEntries.size());

            for (uint_fast64_t state = 0; state < this->originalMatrix.getRowGroupCount(); ++state) {
                for (uint_fast64_t l = 0; l < l_max; ++ l) {
                    for (const auto &keyValue : this->windowVector[state][l]) {
                        unfoldingStates[keyValue.second].state = state;
                        unfoldingStates[keyValue.second].currentValue = keyValue.first;
                        unfoldingStates[keyValue.second].currentWindowSize = l;
                    }
                }
            }
            return unfoldingStates;
        }

        template<typename ValueType>
        uint_fast64_t WindowUnfoldingMeanPayoff<ValueType>::unfoldFrom(
                uint_fast64_t const &state,
                ValueType const &value,
                uint_fast64_t const &l) {

            std::vector<ValueType> const& stateActionRewardsVector = this->rewardModel.getStateActionRewardVector();
            std::vector<uint_fast64_t> const& stateActionIndices = this->originalMatrix.getRowGroupIndices();

            // Initialization
            if (this->newRowGroupEntries.empty()) {
                // the index 0 is the index of the sink state corresponding to states (s, w, l) where l >= l_max and w < 0
                this->newRowGroupEntries.emplace_back();
                this->newRowGroupEntries[0].push_back({std::make_pair(0, storm::utility::one<ValueType>())});
            }
            // i is the index of (state, value, l) in the new matrix
            uint_fast64_t i = this->getNewIndex(state, value, l);
            if (!i) {
                // If the unfolding state (state, value, l) does not yet exist, fill newRowGroupEntries accordingly
                i = this->newRowGroupEntries.size();
                this->newRowGroupEntries.emplace_back();
                this->windowVector[state][l][value] = i;
                // as i was not in the map of weights, unfold the EC from the ith state s_i
                ValueType updatedSumOfWeights;
                uint_fast64_t l_new = l + 1;
                for (uint_fast64_t action = this->enabledActions.getNextSetIndex(stateActionIndices[state]);
                        action < stateActionIndices[state + 1];
                        action = this->enabledActions.getNextSetIndex(action + 1)) {
                    updatedSumOfWeights = value + stateActionRewardsVector[action];
                    // add the current action to the new row group entries
                    this->newRowGroupEntries[i].emplace_back();
                    uint_fast64_t newAction = this->newRowGroupEntries[i].size() - 1;
                    if (updatedSumOfWeights >= 0) {
                        // if the sum of weights is >= 0, then the window closes
                        for (const auto &entry : this->originalMatrix.getRow(action)) {
                            uint_fast64_t successorState = entry.getColumn();
                            ValueType p = entry.getValue();
                            // j is the index of successorState
                            uint_fast64_t j = unfoldFrom(successorState, storm::utility::zero<ValueType>(), 0);
                            this->newRowGroupEntries[i][newAction].push_back(std::make_pair(j, p));
                        }
                    } else if (l_new < this->l_max) {
                        for (const auto &entry : this->originalMatrix.getRow(action)) {
                            uint_fast64_t successorState = entry.getColumn();
                            ValueType p = entry.getValue();
                            // j is the index of successorState
                            uint_fast64_t j = unfoldFrom(successorState, updatedSumOfWeights, l_new);
                            this->newRowGroupEntries[i][newAction].push_back(std::make_pair(j, p));
                        }
                    } else {
                        // if updatedSumOfWeights < 0 and l_new = l_max, then the bounded window objective happens with
                        // probability zero and the window remains open.
                        // Then, s_i transits to the sink state with probability one.
                        this->newRowGroupEntries[i][newAction] = {std::make_pair(0, storm::utility::one<ValueType>())};
                    }
                }
            }
            return i;
        }

        template<typename ValueType>
        bool WindowUnfoldingParity<ValueType>::isEven(ValueType const& priority) {
            return storm::utility::isZero(storm::utility::mod<ValueType>(priority, 2));
        }

        template<typename ValueType>
        uint_fast64_t WindowUnfoldingParity<ValueType>::unfoldFrom(
                uint_fast64_t const &state,
                ValueType const &value,
                uint_fast64_t const &l) {

            std::vector<ValueType> const& priority = this->rewardModel.getStateRewardVector();
            std::vector<uint_fast64_t> const& stateActionIndices = this->originalMatrix.getRowGroupIndices();

            // Initialization
            if (this->newRowGroupEntries.empty()) {
                // the index 0 is the index of the sink state corresponding to states (s, c, l)
                // where l >= l_max - 1 and c is odd.
                this->newRowGroupEntries.emplace_back();
                this->newRowGroupEntries[0].push_back({std::make_pair(0, storm::utility::one<ValueType>())});
            }
            // i is the index of (state, value, l) in the new matrix
            uint_fast64_t i = this->getNewIndex(state, value, l);
            if (!i) {
                // If the unfolding state (state, value, l) does not yet exist, fill newRowGroupEntries accordingly
                i = this->newRowGroupEntries.size();
                this->newRowGroupEntries.emplace_back();
                this->windowVector[state][l][value] = i;
                // as i was not in the map of weights, unfold the EC from the ith state s_i
                uint_fast64_t l_new = l + 1;
                for (uint_fast64_t action = this->enabledActions.getNextSetIndex(stateActionIndices[state]);
                     action < stateActionIndices[state + 1];
                     action = this->enabledActions.getNextSetIndex(action + 1)) {
                    // add the current action to the new row group entries
                    this->newRowGroupEntries[i].emplace_back();
                    uint_fast64_t newAction = this->newRowGroupEntries[i].size() - 1;
                    for (const auto &entry : this->originalMatrix.getRow(action)) {
                        uint_fast64_t successorState = entry.getColumn();
                        ValueType p = entry.getValue();
                        ValueType const& updatedPriority = storm::utility::min<ValueType>(value, priority[successorState]);
                        // j is the index of successorState
                        uint_fast64_t j;
                        if (this->isEven(updatedPriority)) {
                            // if the updated priority is even, then the window closes
                            j = unfoldFrom(successorState, priority[successorState], 0);
                        } else if (l_new < this->l_max - 1) {
                            j = unfoldFrom(successorState, updatedPriority, l_new);
                        } else {
                            // if updatedPriority is odd and l_new = l_max - 1, then the bounded window objective happens
                            // with probability zero and the window remains open. Then, s_i transits to the sink state.
                            j = 0;
                        }
                        this->newRowGroupEntries[i][newAction].push_back(std::make_pair(j, p));
                    }
                }
            }
            return i;
        }

        template class WindowUnfolding<double>;
        template class WindowUnfolding<storm::RationalNumber>;
        template class WindowUnfoldingMeanPayoff<double>;
        template class WindowUnfoldingMeanPayoff<storm::RationalNumber>;
        template class WindowUnfoldingParity<double>;
    }
}
