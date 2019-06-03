//
// Created by Florent Delgrange on 2019-05-24.
//

#include "SchedulerProduct.h"
namespace sw {
    namespace storage {

        template <typename ValueType>
        SchedulerProduct<ValueType>::SchedulerProduct(const storm::models::sparse::Model <ValueType> &sparseModel,
                                                      const storm::storage::Scheduler <ValueType> &scheduler,
                                                      bool forceLabeling)
                                                      : storm::storage::SparseModelMemoryProduct<ValueType>(sparseModel, scheduler),
                                                        isInitialized(false),
                                                        memoryStateCount(scheduler.getNumberOfMemoryStates()),
                                                        model(sparseModel),
                                                        localMemory(scheduler.getMemoryStructure() ? boost::optional<storm::storage::MemoryStructure>() : boost::optional<storm::storage::MemoryStructure>(storm::storage::MemoryStructureBuilder<ValueType>::buildTrivialMemoryStructure(model))),
                                                        memory(scheduler.getMemoryStructure() ? scheduler.getMemoryStructure().get() : localMemory.get()),
                                                        scheduler(scheduler),
                                                        forceLabeling(forceLabeling) {
            reachableStates = storm::storage::BitVector(model.getNumberOfStates() * memoryStateCount, false);
        }

        template <typename ValueType>
        std::shared_ptr<storm::models::sparse::Model<ValueType>> SchedulerProduct<ValueType>::build() {

            initialize();

            // Build the model components
            storm::storage::SparseMatrix<ValueType> transitionMatrix;
            transitionMatrix = buildTransitionMatrixForScheduler();

            storm::models::sparse::StateLabeling stateLabeling = buildStateLabeling(transitionMatrix);
            storm::models::sparse::ChoiceLabeling choiceLabeling = buildChoiceLabeling(transitionMatrix);
            auto rewardModels = buildRewardModels(transitionMatrix);

            return buildResult(std::move(transitionMatrix), std::move(stateLabeling), std::move(choiceLabeling), std::move(rewardModels));
        }

        template <typename ValueType>
        storm::models::sparse::ChoiceLabeling SchedulerProduct<ValueType>::buildChoiceLabeling(const storm::storage::SparseMatrix <ValueType> &resultTransitionMatrix) {

            storm::models::sparse::ChoiceLabeling resultLabeling(resultTransitionMatrix.getRowCount());

            for (uint64_t modelState = 0; modelState < this->model.getNumberOfStates(); ++ modelState) {
                for (uint64_t memoryState = 0; memoryState < this->memoryStateCount; ++memoryState) {
                    if (this->isStateReachable(modelState, memoryState)) {
                        uint_fast64_t choice = this->model.getTransitionMatrix().getRowGroupIndices()[modelState] + this->scheduler->getChoice(modelState, memoryState).getDeterministicChoice();
                        if (this->model.hasChoiceLabeling() and not this->model.getChoiceLabeling().getLabelsOfChoice(choice).empty()) {
                            for (std::string const& label : this->model.getChoiceLabeling().getLabelsOfChoice(choice)) {
                                if (not resultLabeling.containsLabel(label)) {
                                    resultLabeling.addLabel(label);
                                }
                                resultLabeling.addLabelToChoice(label, this->getResultState(modelState, memoryState));
                            }
                        } else {
                            std::ostringstream stream;
                            stream << "a" << choice;
                            std::string label = stream.str();
                            if (not resultLabeling.containsLabel(label)) {
                                resultLabeling.addLabel(label);
                            }
                            resultLabeling.addLabelToChoice(label, this->getResultState(modelState, memoryState));
                        }
                    }
                }
            }

            return resultLabeling;
        }

        template <typename ValueType>
        std::shared_ptr<storm::models::sparse::Model<ValueType>> SchedulerProduct<ValueType>::buildResult(
                storm::storage::SparseMatrix <ValueType> &&matrix, storm::models::sparse::StateLabeling &&stateLabeling,
                storm::models::sparse::ChoiceLabeling &&choiceLabeling,
                std::unordered_map <std::string, storm::models::sparse::StandardRewardModel<ValueType>> &&rewardModels) {

            storm::storage::sparse::ModelComponents<ValueType> components (std::move(matrix), std::move(stateLabeling), std::move(rewardModels));
            components.choiceLabeling = std::move(choiceLabeling);

            storm::models::ModelType resultType = this->model.getType();
            if (this->scheduler && !this->scheduler->isPartialScheduler()) {
                if (this->model.isOfType(storm::models::ModelType::Mdp)) {
                    resultType = storm::models::ModelType::Dtmc;
                }
            }

            return storm::utility::builder::buildModelFromComponents(resultType, std::move(components));
        }


        template<typename ValueType>
        void SchedulerProduct<ValueType>::initialize() {
            if (!isInitialized) {
                uint64_t modelStateCount = model.getNumberOfStates();

                computeMemorySuccessors();

                // Get the initial states and reachable states. A stateIndex s corresponds to the model state (s / memoryStateCount) and memory state (s % memoryStateCount)
                storm::storage::BitVector initialStates(modelStateCount * memoryStateCount, false);
                auto memoryInitIt = memory.getInitialMemoryStates().begin();
                for (auto const& modelInit : model.getInitialStates()) {
                    initialStates.set(modelInit * memoryStateCount + *memoryInitIt, true);
                    ++memoryInitIt;
                }
                STORM_LOG_ASSERT(memoryInitIt == memory.getInitialMemoryStates().end(), "Unexpected number of initial states.");
                computeReachableStates(initialStates);

                // Compute the mapping to the states of the result
                uint64_t reachableStateCount = 0;
                toResultStateMapping = std::vector<uint64_t> (model.getNumberOfStates() * memoryStateCount, std::numeric_limits<uint64_t>::max());
                for (auto const& reachableState : reachableStates) {
                    toResultStateMapping[reachableState] = reachableStateCount;
                    ++reachableStateCount;
                }

                isInitialized = true;
            }
        }

        template<typename ValueType>
        void SchedulerProduct<ValueType>::computeMemorySuccessors() {
            uint64_t modelTransitionCount = model.getTransitionMatrix().getEntryCount();
            memorySuccessors = std::vector<uint64_t>(modelTransitionCount * memoryStateCount, std::numeric_limits<uint64_t>::max());

            for (uint64_t memoryState = 0; memoryState < memoryStateCount; ++memoryState) {
                for (uint64_t transitionGoal = 0; transitionGoal < memoryStateCount; ++transitionGoal) {
                    auto const& memoryTransition = memory.getTransitionMatrix()[memoryState][transitionGoal];
                    if (memoryTransition) {
                        for (auto const& modelTransitionIndex : memoryTransition.get()) {
                            memorySuccessors[modelTransitionIndex * memoryStateCount + memoryState] = transitionGoal;
                        }
                    }
                }
            }
        }

        template<typename ValueType>
        void SchedulerProduct<ValueType>::computeReachableStates(storm::storage::BitVector const &initialStates) {
            // Explore the reachable states via DFS.
            // A state s on the stack corresponds to the model state (s / memoryStateCount) and memory state (s % memoryStateCount)
            reachableStates |= initialStates;
            if (!reachableStates.full()) {
                std::vector<uint64_t> stack(reachableStates.begin(), reachableStates.end());
                while (!stack.empty()) {
                    uint64_t stateIndex = stack.back();
                    stack.pop_back();
                    uint64_t modelState = stateIndex / memoryStateCount;
                    uint64_t memoryState = stateIndex % memoryStateCount;

                    if (scheduler) {
                        auto choices = scheduler->getChoice(modelState, memoryState).getChoiceAsDistribution();
                        uint64_t groupStart = model.getTransitionMatrix().getRowGroupIndices()[modelState];
                        for (auto const& choice : choices) {
                            STORM_LOG_ASSERT(groupStart + choice.first < model.getTransitionMatrix().getRowGroupIndices()[modelState + 1], "Invalid choice " << choice.first << " at model state " << modelState << ".");
                            auto const& row = model.getTransitionMatrix().getRow(groupStart + choice.first);
                            for (auto modelTransitionIt = row.begin(); modelTransitionIt != row.end(); ++modelTransitionIt) {
                                if (!storm::utility::isZero(modelTransitionIt->getValue())) {
                                    uint64_t successorModelState = modelTransitionIt->getColumn();
                                    uint64_t modelTransitionId = modelTransitionIt - model.getTransitionMatrix().begin();
                                    uint64_t successorMemoryState = memorySuccessors[modelTransitionId * memoryStateCount + memoryState];
                                    uint64_t successorStateIndex = successorModelState * memoryStateCount + successorMemoryState;
                                    if (!reachableStates.get(successorStateIndex)) {
                                        reachableStates.set(successorStateIndex, true);
                                        stack.push_back(successorStateIndex);
                                    }
                                }
                            }
                        }
                    } else {
                        auto const& rowGroup = model.getTransitionMatrix().getRowGroup(modelState);
                        for (auto modelTransitionIt = rowGroup.begin(); modelTransitionIt != rowGroup.end(); ++modelTransitionIt) {
                            if (!storm::utility::isZero(modelTransitionIt->getValue())) {
                                uint64_t successorModelState = modelTransitionIt->getColumn();
                                uint64_t modelTransitionId = modelTransitionIt - model.getTransitionMatrix().begin();
                                uint64_t successorMemoryState = memorySuccessors[modelTransitionId * memoryStateCount + memoryState];
                                uint64_t successorStateIndex = successorModelState * memoryStateCount + successorMemoryState;
                                if (!reachableStates.get(successorStateIndex)) {
                                    reachableStates.set(successorStateIndex, true);
                                    stack.push_back(successorStateIndex);
                                }
                            }
                        }
                    }
                }
            }
        }

        template<typename ValueType>
        storm::storage::SparseMatrix<ValueType> SchedulerProduct<ValueType>::buildTransitionMatrixForScheduler() {
            uint64_t numResStates = reachableStates.getNumberOfSetBits();
            uint64_t numResChoices = 0;
            uint64_t numResTransitions = 0;
            bool hasTrivialNondeterminism = true;
            for (auto const& stateIndex : reachableStates) {
                uint64_t modelState = stateIndex / memoryStateCount;
                uint64_t memoryState = stateIndex % memoryStateCount;
                storm::storage::SchedulerChoice<ValueType> choice = scheduler->getChoice(modelState, memoryState);
                if (choice.isDefined()) {
                    ++numResChoices;
                    if (choice.isDeterministic()) {
                        uint64_t modelRow = model.getTransitionMatrix().getRowGroupIndices()[modelState] + choice.getDeterministicChoice();
                        numResTransitions += model.getTransitionMatrix().getRow(modelRow).getNumberOfEntries();
                    } else {
                        std::set<uint64_t> successors;
                        for (auto const& choiceIndex : choice.getChoiceAsDistribution()) {
                            if (!storm::utility::isZero(choiceIndex.second)) {
                                uint64_t modelRow = model.getTransitionMatrix().getRowGroupIndices()[modelState] + choiceIndex.first;
                                for (auto const& entry : model.getTransitionMatrix().getRow(modelRow)) {
                                    successors.insert(entry.getColumn());
                                }
                            }
                        }
                        numResTransitions += successors.size();
                    }
                } else {
                    uint64_t modelRow = model.getTransitionMatrix().getRowGroupIndices()[modelState];
                    uint64_t groupEnd = model.getTransitionMatrix().getRowGroupIndices()[modelState + 1];
                    hasTrivialNondeterminism = hasTrivialNondeterminism && (groupEnd == modelRow + 1);
                    for (; modelRow < groupEnd; ++modelRow) {
                        ++numResChoices;
                        numResTransitions += model.getTransitionMatrix().getRow(modelRow).getNumberOfEntries();
                    }
                }
            }

            storm::storage::SparseMatrixBuilder<ValueType> builder(numResChoices, numResStates, numResTransitions, true, !hasTrivialNondeterminism, hasTrivialNondeterminism ? 0 : numResStates);
            uint64_t currentRow = 0;
            for (auto const& stateIndex : reachableStates) {
                uint64_t modelState = stateIndex / memoryStateCount;
                uint64_t memoryState = stateIndex % memoryStateCount;
                if (!hasTrivialNondeterminism) {
                    builder.newRowGroup(currentRow);
                }
                storm::storage::SchedulerChoice<ValueType> choice = scheduler->getChoice(modelState, memoryState);
                if (choice.isDefined()) {
                    if (choice.isDeterministic()) {
                        uint64_t modelRowIndex = model.getTransitionMatrix().getRowGroupIndices()[modelState] + choice.getDeterministicChoice();
                        auto const& modelRow = model.getTransitionMatrix().getRow(modelRowIndex);
                        for (auto entryIt = modelRow.begin(); entryIt != modelRow.end(); ++entryIt) {
                            uint64_t transitionId = entryIt - model.getTransitionMatrix().begin();
                            uint64_t successorMemoryState = memorySuccessors[transitionId * memoryStateCount + memoryState];
                            builder.addNextValue(currentRow, this->getResultState(entryIt->getColumn(), successorMemoryState), entryIt->getValue());
                        }
                    } else {
                        std::map<uint64_t, ValueType> transitions;
                        for (auto const& choiceIndex : choice.getChoiceAsDistribution()) {
                            if (!storm::utility::isZero(choiceIndex.second)) {
                                uint64_t modelRowIndex = model.getTransitionMatrix().getRowGroupIndices()[modelState] + choiceIndex.first;
                                auto const& modelRow = model.getTransitionMatrix().getRow(modelRowIndex);
                                for (auto entryIt = modelRow.begin(); entryIt != modelRow.end(); ++entryIt) {
                                    uint64_t transitionId = entryIt - model.getTransitionMatrix().begin();
                                    uint64_t successorMemoryState = memorySuccessors[transitionId * memoryStateCount + memoryState];
                                    ValueType transitionValue = choiceIndex.second * entryIt->getValue();
                                    auto insertionRes = transitions.insert(std::make_pair(this->getResultState(entryIt->getColumn(), successorMemoryState), transitionValue));
                                    if (!insertionRes.second) {
                                        insertionRes.first->second += transitionValue;
                                    }
                                }
                            }
                        }
                        for (auto const& transition : transitions) {
                            builder.addNextValue(currentRow, transition.first, transition.second);
                        }
                    }
                    ++currentRow;
                } else {
                    for (uint64_t modelRowIndex = model.getTransitionMatrix().getRowGroupIndices()[modelState]; modelRowIndex < model.getTransitionMatrix().getRowGroupIndices()[modelState + 1]; ++modelRowIndex) {
                        auto const& modelRow = model.getTransitionMatrix().getRow(modelRowIndex);
                        for (auto entryIt = modelRow.begin(); entryIt != modelRow.end(); ++entryIt) {
                            uint64_t transitionId = entryIt - model.getTransitionMatrix().begin();
                            uint64_t successorMemoryState = memorySuccessors[transitionId * memoryStateCount + memoryState];
                            builder.addNextValue(currentRow, this->getResultState(entryIt->getColumn(), successorMemoryState), entryIt->getValue());
                        }
                        ++currentRow;
                    }
                }
            }

            return builder.build();
        }

        template<typename ValueType>
        storm::models::sparse::StateLabeling SchedulerProduct<ValueType>::buildStateLabeling(
                const storm::storage::SparseMatrix<ValueType> &resultTransitionMatrix) {
            uint64_t modelStateCount = model.getNumberOfStates();

            uint64_t numResStates = resultTransitionMatrix.getRowGroupCount();
            storm::models::sparse::StateLabeling resultLabeling(numResStates);

            for (std::string modelLabel : model.getStateLabeling().getLabels()) {
                if (modelLabel != "init") {
                    storm::storage::BitVector resLabeledStates(numResStates, false);
                    for (auto const& modelState : model.getStateLabeling().getStates(modelLabel)) {
                        for (uint64_t memoryState = 0; memoryState < memoryStateCount; ++memoryState) {
                            if (this->isStateReachable(modelState, memoryState)) {
                                resLabeledStates.set(this->getResultState(modelState, memoryState), true);
                            }
                        }
                    }
                    resultLabeling.addLabel(modelLabel, std::move(resLabeledStates));
                }
            }
            for (std::string memoryLabel : memory.getStateLabeling().getLabels()) {
                STORM_LOG_THROW(!resultLabeling.containsLabel(memoryLabel), storm::exceptions::InvalidOperationException, "Failed to build the product of model and memory structure: State labelings are not disjoint as both structures contain the label " << memoryLabel << ".");
                storm::storage::BitVector resLabeledStates(numResStates, false);
                for (auto const& memoryState : memory.getStateLabeling().getStates(memoryLabel)) {
                    for (uint64_t modelState = 0; modelState < modelStateCount; ++modelState) {
                        if (this->isStateReachable(modelState, memoryState)) {
                            resLabeledStates.set(this->getResultState(modelState, memoryState), true);
                        }
                    }
                }
                resultLabeling.addLabel(memoryLabel, std::move(resLabeledStates));
            }

            storm::storage::BitVector initialStates(numResStates, false);
            auto memoryInitIt = memory.getInitialMemoryStates().begin();
            for (auto const& modelInit : model.getInitialStates()) {
                initialStates.set(this->getResultState(modelInit, *memoryInitIt), true);
                ++memoryInitIt;
            }
            resultLabeling.addLabel("init", std::move(initialStates));

            return resultLabeling;
        }

        template <typename ValueType>
        std::unordered_map<std::string, storm::models::sparse::StandardRewardModel<ValueType>> SchedulerProduct<ValueType>::buildRewardModels(storm::storage::SparseMatrix<ValueType> const& resultTransitionMatrix) {

            typedef typename storm::models::sparse::StandardRewardModel<ValueType> RewardModelType;

            std::unordered_map<std::string, RewardModelType> result;
            uint64_t numResStates = resultTransitionMatrix.getRowGroupCount();

            for (auto const& rewardModel : model.getRewardModels()) {
                boost::optional<std::vector<ValueType>> stateRewards;
                if (rewardModel.second.hasStateRewards()) {
                    stateRewards = std::vector<ValueType>(numResStates, storm::utility::zero<ValueType>());
                    uint64_t modelState = 0;
                    for (auto const& modelStateReward : rewardModel.second.getStateRewardVector()) {
                        if (!storm::utility::isZero(modelStateReward)) {
                            for (uint64_t memoryState = 0; memoryState < memoryStateCount; ++memoryState) {
                                if (this->isStateReachable(modelState, memoryState)) {
                                    stateRewards.get()[this->getResultState(modelState, memoryState)] = modelStateReward;
                                }
                            }
                        }
                        ++modelState;
                    }
                }
                boost::optional<std::vector<ValueType>> stateActionRewards;
                if (rewardModel.second.hasStateActionRewards()) {
                    stateActionRewards = std::vector<ValueType>(resultTransitionMatrix.getRowCount(), storm::utility::zero<ValueType>());
                    uint64_t modelState = 0;
                    uint64_t modelRow = 0;
                    for (auto const& modelStateActionReward : rewardModel.second.getStateActionRewardVector()) {
                        if (!storm::utility::isZero(modelStateActionReward)) {
                            while (modelRow >= model.getTransitionMatrix().getRowGroupIndices()[modelState + 1]) {
                                ++modelState;
                            }
                            uint64_t rowOffset = modelRow - model.getTransitionMatrix().getRowGroupIndices()[modelState];
                            for (uint64_t memoryState = 0; memoryState < memoryStateCount; ++memoryState) {
                                if (this->isStateReachable(modelState, memoryState)) {
                                    if (scheduler && scheduler->getChoice(modelState, memoryState).isDefined()) {
                                        ValueType factor = scheduler->getChoice(modelState, memoryState).getChoiceAsDistribution().getProbability(rowOffset);
                                        stateActionRewards.get()[resultTransitionMatrix.getRowGroupIndices()[this->getResultState(modelState, memoryState)]] = factor * modelStateActionReward;
                                    } else {
                                        stateActionRewards.get()[resultTransitionMatrix.getRowGroupIndices()[this->getResultState(modelState, memoryState)] + rowOffset] = modelStateActionReward;
                                    }
                                }
                            }
                        }
                        ++modelRow;
                    }
                }
                boost::optional<storm::storage::SparseMatrix<ValueType>> transitionRewards;
                if (rewardModel.second.hasTransitionRewards()) {
                    storm::storage::SparseMatrixBuilder<ValueType> builder(resultTransitionMatrix.getRowCount(), resultTransitionMatrix.getColumnCount());
                    uint64_t stateIndex = 0;
                    for (auto const& resState : toResultStateMapping) {
                        if (resState < numResStates) {
                            uint64_t modelState = stateIndex / memoryStateCount;
                            uint64_t memoryState = stateIndex % memoryStateCount;
                            uint64_t rowGroupSize = resultTransitionMatrix.getRowGroupSize(resState);
                            if (scheduler && scheduler->getChoice(modelState, memoryState).isDefined()) {
                                std::map<uint64_t, ValueType> rewards;
                                for (uint64_t rowOffset = 0; rowOffset < rowGroupSize; ++rowOffset) {
                                    uint64_t modelRowIndex = model.getTransitionMatrix().getRowGroupIndices()[modelState] + rowOffset;
                                    auto transitionEntryIt = model.getTransitionMatrix().getRow(modelRowIndex).begin();
                                    for (auto const& rewardEntry : rewardModel.second.getTransitionRewardMatrix().getRow(modelRowIndex)) {
                                        while (transitionEntryIt->getColumn() != rewardEntry.getColumn()) {
                                            STORM_LOG_ASSERT(transitionEntryIt != model.getTransitionMatrix().getRow(modelRowIndex).end(), "The reward transition matrix is not a submatrix of the model transition matrix.");
                                            ++transitionEntryIt;
                                        }
                                        uint64_t transitionId = transitionEntryIt - model.getTransitionMatrix().begin();
                                        uint64_t successorMemoryState = memorySuccessors[transitionId * memoryStateCount + memoryState];
                                        auto insertionRes = rewards.insert(std::make_pair(this->getResultState(rewardEntry.getColumn(), successorMemoryState), rewardEntry.getValue()));
                                        if (!insertionRes.second) {
                                            insertionRes.first->second += rewardEntry.getValue();
                                        }
                                    }
                                }
                                uint64_t resRowIndex = resultTransitionMatrix.getRowGroupIndices()[resState];
                                for (auto& reward : rewards) {
                                    builder.addNextValue(resRowIndex, reward.first, reward.second);
                                }
                            } else {
                                for (uint64_t rowOffset = 0; rowOffset < rowGroupSize; ++rowOffset) {
                                    uint64_t resRowIndex = resultTransitionMatrix.getRowGroupIndices()[resState] + rowOffset;
                                    uint64_t modelRowIndex = model.getTransitionMatrix().getRowGroupIndices()[modelState] + rowOffset;
                                    auto transitionEntryIt = model.getTransitionMatrix().getRow(modelRowIndex).begin();
                                    for (auto const& rewardEntry : rewardModel.second.getTransitionRewardMatrix().getRow(modelRowIndex)) {
                                        while (transitionEntryIt->getColumn() != rewardEntry.getColumn()) {
                                            STORM_LOG_ASSERT(transitionEntryIt != model.getTransitionMatrix().getRow(modelRowIndex).end(), "The reward transition matrix is not a submatrix of the model transition matrix.");
                                            ++transitionEntryIt;
                                        }
                                        uint64_t transitionId = transitionEntryIt - model.getTransitionMatrix().begin();
                                        uint64_t successorMemoryState = memorySuccessors[transitionId * memoryStateCount + memoryState];
                                        builder.addNextValue(resRowIndex, this->getResultState(rewardEntry.getColumn(), successorMemoryState), rewardEntry.getValue());
                                    }
                                }
                            }
                        }
                        ++stateIndex;
                    }
                    transitionRewards = builder.build();
                }
                result.insert(std::make_pair(rewardModel.first, RewardModelType(std::move(stateRewards), std::move(stateActionRewards), std::move(transitionRewards))));
            }
            return result;
        }

        template class SchedulerProduct<double>;
        template class SchedulerProduct<storm::RationalNumber>;
        template class SchedulerProduct<storm::RationalFunction>;
    }
}
