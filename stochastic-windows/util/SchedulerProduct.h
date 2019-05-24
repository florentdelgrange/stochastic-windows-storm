//
// Created by Florent Delgrange on 2019-05-24.
//

#ifndef STORM_SCHEDULERPRODUCT_H
#define STORM_SCHEDULERPRODUCT_H

#include <storm/storage/memorystructure/SparseModelMemoryProduct.h>
#include <storm/models/sparse/MarkovAutomaton.h>
#include <storm/utility/builder.h>

namespace sw {
    namespace storage {

        template <typename ValueType>
        class SchedulerProduct : public storm::storage::SparseModelMemoryProduct<ValueType> {
        public:

            SchedulerProduct(storm::models::sparse::Model<ValueType> const& sparseModel, storm::storage::Scheduler<ValueType> const& scheduler);

            std::shared_ptr<storm::models::sparse::Model<ValueType>> build();

        protected:

            storm::models::sparse::ChoiceLabeling buildChoiceLabeling(storm::storage::SparseMatrix<ValueType> const& resultTransitionMatrix);
            std::shared_ptr<storm::models::sparse::Model<ValueType>> buildResult(storm::storage::SparseMatrix<ValueType>&& matrix, storm::models::sparse::StateLabeling&& stateLabeling, storm::models::sparse::ChoiceLabeling&& choiceLabeling, std::unordered_map <std::string, storm::models::sparse::StandardRewardModel<ValueType>>&& rewardModels);

        };

    }
}


#endif //STORM_SCHEDULERPRODUCT_H
