#include <memory>

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
                  originalModel(mdp),
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
                : WindowUnfolding<ValueType>(mdp, rewardModelName, l_max, storm::storage::BitVector(mdp.getNumberOfChoices(), true)) {}


        template<typename ValueType>
        WindowUnfoldingMeanPayoff<ValueType>::WindowUnfoldingMeanPayoff(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const &rewardModelName, uint_fast64_t const &l_max,
                storm::storage::BitVector const& initialStates,
                storm::storage::BitVector const& enabledActions)
                : WindowUnfolding<ValueType>(mdp, rewardModelName, l_max, enabledActions) {
            assert(initialStates.size() == mdp.getNumberOfStates());
            this->generateMatrix(initialStates);
        }

        template<typename ValueType>
        WindowUnfoldingMeanPayoff<ValueType>::WindowUnfoldingMeanPayoff(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const &rewardModelName, uint_fast64_t const &l_max,
                storm::storage::BitVector const &initialStates)
                : WindowUnfolding<ValueType>(mdp, rewardModelName, l_max) {
            assert(initialStates.size() == mdp.getNumberOfStates());
            this->generateMatrix(initialStates);
        }

        template<typename ValueType>
        WindowUnfoldingParity<ValueType>::WindowUnfoldingParity(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const &rewardModelName, uint_fast64_t const &l_max,
                storm::storage::BitVector const& initialStates,
                storm::storage::BitVector const& enabledActions)
                : WindowUnfolding<ValueType>(mdp, rewardModelName, l_max, enabledActions) {
            assert(initialStates.size() == mdp.getNumberOfStates());
            this->generateMatrix(initialStates);
        }

        template<typename ValueType>
        WindowUnfoldingParity<ValueType>::WindowUnfoldingParity(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const &rewardModelName, uint_fast64_t const &l_max,
                storm::storage::BitVector const &initialStates)
                : WindowUnfolding<ValueType>(mdp, rewardModelName, l_max) {
            assert(initialStates.size() == mdp.getNumberOfStates());
            this->generateMatrix(initialStates);
        }

        template<typename ValueType>
        void WindowUnfolding<ValueType>::generateMatrix(storm::storage::BitVector const &initialStates) {
            // Unfold the MDP from the initial states
            for (auto state: initialStates) {
                unfoldFrom(state, initialStateValue(state), 0);
            }
            // this line allows to have the same number of columns and rowGroups in the sparse matrix
            this->newRowGroupEntries[0][0].push_back(std::make_pair(newRowGroupEntries.size() - 1, storm::utility::zero<ValueType>()));
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
        ValueType WindowUnfoldingMeanPayoff<ValueType>::initialStateValue(uint_fast64_t initialState) const {
            return storm::utility::zero<ValueType>();
        }

        template<typename ValueType>
        ValueType WindowUnfoldingParity<ValueType>::initialStateValue(uint_fast64_t initialState) const {
            return this->rewardModel.getStateReward(initialState);
        }

        template<typename ValueType>
        uint_fast64_t WindowUnfolding<ValueType>::getNewIndex(uint_fast64_t state, ValueType value, uint_fast64_t currentWindowSize) const {

            auto keyValue = this->windowVector[state][currentWindowSize].find(value);
            if (keyValue == this->windowVector[state][currentWindowSize].end()) {
                // 0 is a special value (it represents the sink state in the unfolding);
                // it means that the input (state, value, currentWindowSize) does not yet exist in the unfolding.
                return 0;
            } else {
                return keyValue->second;
            }
        }

        template <typename ValueType>
        WindowMemory<ValueType> WindowUnfolding<ValueType>::generateMemory(bool setLabels) const {

            WindowMemory<ValueType> windowMemory;
            windowMemory.unfoldingToMemoryStatesMapping = std::vector<uint_fast64_t>(this->matrix.getRowGroupCount());
            std::vector<std::unordered_map<ValueType, uint_fast64_t>> windowSizeValueMapping(this->l_max);
            for (uint_fast64_t l = 0; l < l_max; ++ l) {
                windowSizeValueMapping[l] = std::unordered_map<ValueType, uint_fast64_t>();
            }

            // keeps track of the current memory of each state during the memory assignment
            std::vector<uint_fast64_t> currentMemory(this->originalModel.getNumberOfStates(), 0);
            // mappings between the unfolding and the original MDP
            std::vector<StateValueWindowSize<ValueType>> unfoldingStatesMeaning = this->getNewStatesMeaning();
            std::vector<uint_fast64_t> unfoldingActionsMeaning = this->newToOldActionsMapping(unfoldingStatesMeaning);

            // assign a memory index to each state of the unfolding
            uint_fast64_t memory = 0;
            for (uint_fast64_t unfoldingState = 1; unfoldingState < this->matrix.getRowGroupCount(); ++ unfoldingState) {
                StateValueWindowSize<ValueType> const& stateValueWindowSize = unfoldingStatesMeaning[unfoldingState];
                auto keyValue = windowSizeValueMapping[stateValueWindowSize.currentWindowSize].find(stateValueWindowSize.currentValue);
                if (keyValue == windowSizeValueMapping[stateValueWindowSize.currentWindowSize].end()) {
                    windowSizeValueMapping[stateValueWindowSize.currentWindowSize][stateValueWindowSize.currentValue] = memory;
                    ++ memory;
                }
                windowMemory.unfoldingToMemoryStatesMapping[unfoldingState] =
                        windowSizeValueMapping[stateValueWindowSize.currentWindowSize][stateValueWindowSize.currentValue];
            }

            uint_fast64_t M = memory;
            storm::storage::MemoryStructureBuilder<ValueType> memoryBuilder(M + 1, this->originalModel);
            // the last state of the memory structure is the sink state corresponding to the windows staying open for l_max steps
            windowMemory.unfoldingToMemoryStatesMapping[0] = M;
            if (setLabels){
                memoryBuilder.setLabel(M, "⊥");
                for (uint_fast64_t l = 0; l < this->l_max; ++ l) {
                    for (const auto& keyValue : windowSizeValueMapping[l]) {
                        // memory label is (current value, current window size)
                        std::ostringstream stream;
                        stream << "(" << boost::lexical_cast<std::string>(keyValue.first) << ", "
                               << std::to_string(l) << ")";
                        memoryBuilder.setLabel(keyValue.second, stream.str());
                    }
                }
            }

            storm::storage::MemoryStructure::TransitionMatrix
                 // contains the memory transitions m -> m' where -> is triggered when the original MDP transits to a subset of states
                 unfoldingStatesMemoryTransitions(M, std::vector<boost::optional<storm::storage::BitVector>>(M + 1)),
                 // contains the memory transitions m -> m' where -> is triggered when a subset of actions is chosen in the original MDP
                 unfoldingActionsMemoryTransitions(M, std::vector<boost::optional<storm::storage::BitVector>>(M + 1)),
                 // contains the memory transitions m -> M for each a (states for which m transit to M when a is chosen)
                 sinkTransitions(M, std::vector<boost::optional<storm::storage::BitVector>>(this->originalModel.getNumberOfChoices()));
            // initialization of transitions to the sink state
            for (memory = 0; memory < M; ++ memory) {
                for (uint_fast64_t action = 0; action < this->originalModel.getNumberOfChoices(); ++ action) {
                    sinkTransitions[memory][action] = storm::storage::BitVector(this->originalModel.getNumberOfStates(), true);
                }
            }
            // computes the memory structure
            for (uint_fast64_t unfoldingState = 1; unfoldingState < this->matrix.getRowGroupCount(); ++ unfoldingState) {
                memory = windowMemory.unfoldingToMemoryStatesMapping[unfoldingState];
                for (uint_fast64_t unfoldingAction = this->matrix.getRowGroupIndices()[unfoldingState];
                     unfoldingAction < this->matrix.getRowGroupIndices()[unfoldingState + 1];
                     ++ unfoldingAction) {
                    for (const auto &entry : this->matrix.getRow(unfoldingAction)) {
                        uint_fast64_t unfoldingSuccessor = entry.getColumn();
                        uint_fast64_t next_memory = windowMemory.unfoldingToMemoryStatesMapping[unfoldingSuccessor];
                        if (unfoldingSuccessor and not unfoldingStatesMemoryTransitions[memory][next_memory]) {
                            unfoldingActionsMemoryTransitions[memory][next_memory] = storm::storage::BitVector(this->originalModel.getNumberOfChoices());
                            unfoldingStatesMemoryTransitions[memory][next_memory] = storm::storage::BitVector(this->originalModel.getNumberOfStates());
                        }
                        if (unfoldingSuccessor) {
                            unfoldingActionsMemoryTransitions[memory][next_memory]->set(unfoldingActionsMeaning[unfoldingAction], true);
                            unfoldingStatesMemoryTransitions[memory][next_memory]->set(unfoldingStatesMeaning[unfoldingSuccessor].state, true);
                            sinkTransitions[memory][unfoldingActionsMeaning[unfoldingAction]]->set(unfoldingStatesMeaning[unfoldingSuccessor].state, false);
                        }
                    }
                }
            }

            for (memory = 0; memory < M; ++ memory) {
                for (uint_fast64_t next_memory = 0; next_memory < M; ++ next_memory) {
                    if (unfoldingActionsMemoryTransitions[memory][next_memory]) {
                        memoryBuilder.setTransition(memory, next_memory,
                                                    *unfoldingStatesMemoryTransitions[memory][next_memory],
                                                    unfoldingActionsMemoryTransitions[memory][next_memory]);
                    }
                }
                for (uint_fast64_t action = 0; action < this->originalModel.getNumberOfChoices(); ++ action) {
                    memoryBuilder.setTransition(memory, M, *sinkTransitions[memory][action], action);
                }
            }
            memoryBuilder.setTransition(M, M, storm::storage::BitVector(this->originalModel.getNumberOfStates(), true));
            // initial memory states
            for (uint_fast64_t const& originalState : this->originalModel.getInitialStates()) {
               memoryBuilder.setInitialMemoryState(originalState, windowMemory.unfoldingToMemoryStatesMapping[this->getInitialState(originalState)]);
            }
            storm::storage::MemoryStructure memoryStructure = memoryBuilder.build();
            windowMemory.memoryStructure = std::make_unique<storm::storage::MemoryStructure>(std::move(memoryStructure));

            return windowMemory;
        }

        template<typename ValueType>
        uint_fast64_t WindowUnfoldingMeanPayoff<ValueType>::getInitialState(uint_fast64_t originalInitialState) const {
            return this->getNewIndex(originalInitialState, this->initialStateValue(originalInitialState), 0);
        }

        template<typename ValueType>
        uint_fast64_t WindowUnfoldingParity<ValueType>::getInitialState(uint_fast64_t originalInitialState) const {
            return this->getNewIndex(originalInitialState, this->initialStateValue(originalInitialState), 0);
        }

        template<typename ValueType>
        storm::storage::SparseMatrix<ValueType> const& WindowUnfolding<ValueType>::getMatrix() const {
            return this->matrix;
        }

        template<typename ValueType>
        std::vector<StateValueWindowSize<ValueType>>
        WindowUnfolding<ValueType>::getNewStatesMeaning() const {

            std::vector<StateValueWindowSize<ValueType>> unfoldingStates(
                    this->newRowGroupEntries.size());

            for (uint_fast64_t state = 0; state < this->originalMatrix.getRowGroupCount(); ++ state) {
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

        template <typename ValueType>
        std::vector<uint_fast64_t> WindowUnfolding<ValueType>::newToOldActionsMapping(std::vector<StateValueWindowSize<ValueType>> const& newStatesMeaning) const {

            std::vector<uint_fast64_t> unfoldingActionsMeaning(this->matrix.getRowCount());
            for (uint_fast64_t unfoldingState = 1; unfoldingState < this->matrix.getRowGroupCount(); ++ unfoldingState) {
                uint_fast64_t originalState = newStatesMeaning[unfoldingState].state;
                uint_fast64_t unfoldingAction = this->matrix.getRowGroupIndices()[unfoldingState];
                for (uint_fast64_t originalAction = this->enabledActions.getNextSetIndex(this->originalMatrix.getRowGroupIndices()[originalState]);
                     originalAction < this->originalMatrix.getRowGroupIndices()[originalState + 1];
                     originalAction = this->enabledActions.getNextSetIndex(originalAction + 1)) {
                    unfoldingActionsMeaning[unfoldingAction] = originalAction;
                    ++ unfoldingAction;
                }
                assert(unfoldingAction == this->matrix.getRowGroupIndices()[unfoldingState + 1]);
            }
            return unfoldingActionsMeaning;
        }

        template<typename ValueType>
        storm::storage::BitVector const &WindowUnfolding<ValueType>::getOriginalEnabledActions() const {
            return this->enabledActions;
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
