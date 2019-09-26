//
// Created by Florent Delgrange on 2019-05-24.
//

#ifndef STORM_SCHEDULERPRODUCT_H
#define STORM_SCHEDULERPRODUCT_H

#include <storm/storage/memorystructure/SparseModelMemoryProduct.h>
#include <storm/storage/memorystructure/MemoryStructureBuilder.h>
#include <storm/utility/builder.h>

namespace sw {
    namespace storage {

        struct SchedulerProductLabeling {
            explicit SchedulerProductLabeling(boost::optional<std::string> &&weights = boost::none, boost::optional<std::string> &&priorities = boost::none)
            : weights(std::move(weights)), priorities(std::move(priorities)) {}
            boost::optional<std::string> weights;
            boost::optional<std::string> priorities;
        };

        template <typename ValueType>
        class SchedulerProduct : public storm::storage::SparseModelMemoryProduct<ValueType> {
        public:

            SchedulerProduct(storm::models::sparse::Model<ValueType> const& sparseModel,
                             storm::storage::Scheduler<ValueType> const& scheduler,
                             SchedulerProductLabeling const& labelingOptions);

            // Invokes the building of the product under the specified scheduler (if given).
            std::shared_ptr<storm::models::sparse::Model<ValueType>> build();

        private:

            // Initializes auxiliary data for building the product
            void initialize();

            // Computes for each pair of memory state and model transition index the successor memory state
            // The resulting vector maps (modelTransition * memoryStateCount) + memoryState to the corresponding successor state of the memory structure
            void computeMemorySuccessors();

            // Computes the reachable states of the resulting model
            void computeReachableStates(storm::storage::BitVector const& initialStates);

            // Method that build the model components
            // Matrix for models that consider a scheduler
            storm::storage::SparseMatrix<ValueType> buildTransitionMatrixForScheduler();
            // State labeling.
            storm::models::sparse::StateLabeling buildStateLabeling(storm::storage::SparseMatrix<ValueType> const& resultTransitionMatrix);
            // Choice labeling
            storm::models::sparse::ChoiceLabeling buildChoiceLabeling(storm::storage::SparseMatrix<ValueType> const& resultTransitionMatrix);
            // Reward models
            std::unordered_map<std::string, storm::models::sparse::StandardRewardModel<ValueType>> buildRewardModels(storm::storage::SparseMatrix<ValueType> const& resultTransitionMatrix);
            // Builds the resulting model
            std::shared_ptr<storm::models::sparse::Model<ValueType>> buildResult(storm::storage::SparseMatrix<ValueType>&& matrix, storm::models::sparse::StateLabeling&& stateLabeling, storm::models::sparse::ChoiceLabeling&& choiceLabeling, std::unordered_map <std::string, storm::models::sparse::StandardRewardModel<ValueType>>&& rewardModels);

            // Stores whether this builder has already been initialized.
            bool isInitialized;

            // stores the successor memory states for each transition in the product
            std::vector<uint64_t> memorySuccessors;

            // Maps (modelState * memoryStateCount) + memoryState to the state in the result that represents (memoryState,modelState)
            std::vector<uint64_t> toResultStateMapping;

            // Indicates which states are considered reachable. (s, m) is reachable if this BitVector is true at (s * memoryStateCount) + m
            storm::storage::BitVector reachableStates;

            uint64_t const memoryStateCount;

            storm::models::sparse::Model<ValueType> const& model;
            boost::optional<storm::storage::MemoryStructure> localMemory;
            storm::storage::MemoryStructure const& memory;
            boost::optional<storm::storage::Scheduler<ValueType> const&>  scheduler;
            SchedulerProductLabeling labelingOptions;

        };

    }
}


#endif //STORM_SCHEDULERPRODUCT_H
