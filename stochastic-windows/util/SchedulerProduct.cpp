//
// Created by Florent Delgrange on 2019-05-24.
//

#include "SchedulerProduct.h"
namespace sw {
    namespace storage {

        template <typename ValueType>
        SchedulerProduct<ValueType>::SchedulerProduct(const storm::models::sparse::Model <ValueType> &sparseModel,
                                                      const storm::storage::Scheduler <ValueType> &scheduler)
                                                      : storm::storage::SparseModelMemoryProduct<ValueType>(sparseModel, scheduler) {}

        template <typename ValueType>
        std::shared_ptr<storm::models::sparse::Model<ValueType>> SchedulerProduct<ValueType>::build() {

            this->initialize();

            // Build the model components
            storm::storage::SparseMatrix<ValueType> transitionMatrix;
            if (this->scheduler) {
                transitionMatrix = this->buildTransitionMatrixForScheduler();
            } else if (this->model.getTransitionMatrix().hasTrivialRowGrouping()) {
                transitionMatrix = this->buildDeterministicTransitionMatrix();
            } else {
                transitionMatrix = this->buildNondeterministicTransitionMatrix();
            }
            storm::models::sparse::StateLabeling stateLabeling = this->buildStateLabeling(transitionMatrix);
            storm::models::sparse::ChoiceLabeling choiceLabeling = buildChoiceLabeling(transitionMatrix);
            auto rewardModels = this->buildRewardModels(transitionMatrix);

            return this->buildResult(std::move(transitionMatrix), std::move(stateLabeling), std::move(choiceLabeling), std::move(rewardModels));
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

            if (this->model.isOfType(storm::models::ModelType::Ctmc)) {
                components.rateTransitions = true;
            } else if (this->model.isOfType(storm::models::ModelType::MarkovAutomaton)) {
                // We also need to translate the exit rates and the Markovian states
                uint64_t numResStates = components.transitionMatrix.getRowGroupCount();
                std::vector<ValueType> resultExitRates;
                resultExitRates.reserve(components.transitionMatrix.getRowGroupCount());
                storm::storage::BitVector resultMarkovianStates(numResStates, false);
                auto const& modelExitRates = dynamic_cast<storm::models::sparse::MarkovAutomaton<ValueType> const&>(this->model).getExitRates();
                auto const& modelMarkovianStates = dynamic_cast<storm::models::sparse::MarkovAutomaton<ValueType> const&>(this->model).getMarkovianStates();

                uint64_t stateIndex = 0;
                for (auto const& resState : this->toResultStateMapping) {
                    if (resState < numResStates) {
                        assert(resState == resultExitRates.size());
                        uint64_t modelState = stateIndex / this->memoryStateCount;
                        resultExitRates.push_back(modelExitRates[modelState]);
                        if (modelMarkovianStates.get(modelState)) {
                            resultMarkovianStates.set(resState, true);
                        }
                    }
                    ++stateIndex;
                }
                components.markovianStates = std::move(resultMarkovianStates);
                components.exitRates = std::move(resultExitRates);
            }

            storm::models::ModelType resultType = this->model.getType();
            if (this->scheduler && !this->scheduler->isPartialScheduler()) {
                if (this->model.isOfType(storm::models::ModelType::Mdp)) {
                    resultType = storm::models::ModelType::Dtmc;
                }
                // Note that converting deterministic MAs to CTMCs via state elimination does not preserve all properties (e.g. step bounded)
            }

            return storm::utility::builder::buildModelFromComponents(resultType, std::move(components));
        }

        template class SchedulerProduct<double>;
        template class SchedulerProduct<storm::RationalNumber>;
        template class SchedulerProduct<storm::RationalFunction>;
    }
}
