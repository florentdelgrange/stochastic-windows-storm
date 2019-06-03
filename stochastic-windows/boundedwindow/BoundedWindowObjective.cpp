//
// Created by Florent Delgrange on 2019-03-26.
//

#include "BoundedWindowObjective.h"

namespace sw {
    namespace BoundedWindow {

        template <typename ValueType>
        BoundedWindowObjective<ValueType>::BoundedWindowObjective(
                const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                std::string const &rewardModelName,
                ClassificationMethod classificationMethod)
                : WindowObjective<ValueType>(mdp, rewardModelName), classificationMethod(classificationMethod){}

        template <typename ValueType>
        BoundedWindowMeanPayoffObjective<ValueType>::BoundedWindowMeanPayoffObjective(
                const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                std::string const &rewardModelName,
                ClassificationMethod classificationMethod)
                : BoundedWindowObjective<ValueType>(mdp, rewardModelName, classificationMethod) {}

        template <typename ValueType>
        BoundedWindowParityObjective<ValueType>::BoundedWindowParityObjective(
                const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                std::string const &rewardModelName,
                ClassificationMethod classificationMethod)
                : BoundedWindowObjective<ValueType>(mdp, rewardModelName, classificationMethod) {}

        template <typename ValueType>
        storm::storage::BitVector BoundedWindowMeanPayoffObjective<ValueType>::getGoodStateSpace() const {
            std::unique_ptr<sw::util::MaximalEndComponentClassifier<ValueType>> mecClassifier;
            switch (this->classificationMethod) {
                case MemorylessWindowGame: {
                    sw::storage::MaximalEndComponentDecompositionWindowMeanPayoffGame<ValueType> mecGames(this->mdp, this->rewardModelName);
                    mecClassifier = std::unique_ptr<MaximalEndComponentClassifier<ValueType>>(
                            new MaximalEndComponentClassifier<ValueType>(this->mdp, mecGames)
                    );
                } break;
                case WindowGameWithBound: {
                    sw::storage::MaximalEndComponentDecompositionWindowMeanPayoffGame<ValueType>
                    mecGames(this->mdp, this->rewardModelName, this->getUniformBound());
                    mecClassifier = std::unique_ptr<sw::FixedWindow::MaximalEndComponentClassifier<ValueType>>(
                            new sw::FixedWindow::MaximalEndComponentClassifier<ValueType>(this->mdp, mecGames)
                    );
                } break;
                case Unfolding: {
                    sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<ValueType>
                    unfoldedMECs(this->mdp, this->rewardModelName, this->getUniformBound());
                    mecClassifier = std::unique_ptr<sw::FixedWindow::MaximalEndComponentClassifier<ValueType>>(
                            new sw::FixedWindow::MaximalEndComponentClassifier<ValueType>(this->mdp, unfoldedMECs)
                    );
                } break;
            }
            return std::move(mecClassifier->getGoodStateSpace());
        }

        template<typename ValueType>
        uint_fast64_t BoundedWindowMeanPayoffObjective<ValueType>::getUniformBound() const {
            std::vector<ValueType> const& weights = this->mdp.getRewardModel(this->rewardModelName).getStateActionRewardVector();
            ValueType W = -1 * storm::utility::infinity<ValueType>();
            storm::storage::MaximalEndComponentDecomposition<ValueType> mecDecomposition(this->mdp);
            {
                ValueType absoluteWeight;
                for (storm::storage::MaximalEndComponent const& mec : mecDecomposition) {
                    for (auto const& stateActionsPair : mec) {
                        for (auto const& action : stateActionsPair.second) {
                            absoluteWeight = storm::utility::abs(weights[action]);
                            if (W < absoluteWeight) {
                                W = absoluteWeight;
                            }
                        }
                    }
                }
            }
            auto W_int = storm::utility::convertNumber<uint_fast64_t>(storm::utility::ceil(W));
            return (this->mdp.getNumberOfStates() - 1) * (this->mdp.getNumberOfStates() * W_int + 1);
        }

        template<typename ValueType>
        storage::GoodStateSpaceAndScheduler<ValueType>
        BoundedWindowMeanPayoffObjective<ValueType>::produceGoodScheduler(bool memoryStatesLabeling) const {
            std::unique_ptr<sw::util::MaximalEndComponentClassifier<ValueType>> mecClassifier;
            switch (this->classificationMethod) {
                case MemorylessWindowGame: {
                    sw::storage::MaximalEndComponentDecompositionWindowMeanPayoffGame<ValueType> mecGames(this->mdp, this->rewardModelName);
                    mecClassifier = std::unique_ptr<MaximalEndComponentClassifier<ValueType>>(
                            new MaximalEndComponentClassifier<ValueType>(this->mdp, mecGames, true)
                    );
                } break;
                case WindowGameWithBound: {
                    sw::storage::MaximalEndComponentDecompositionWindowMeanPayoffGame<ValueType>
                            mecGames(this->mdp, this->rewardModelName, this->getUniformBound());
                    mecClassifier = std::unique_ptr<sw::FixedWindow::MaximalEndComponentClassifier<ValueType>>(
                            new sw::FixedWindow::MaximalEndComponentClassifier<ValueType>(this->mdp, mecGames, true, memoryStatesLabeling)
                    );
                } break;
                case Unfolding: {
                    sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<ValueType>
                            unfoldedMECs(this->mdp, this->rewardModelName, this->getUniformBound());
                    mecClassifier = std::unique_ptr<sw::FixedWindow::MaximalEndComponentClassifier<ValueType>>(
                            new sw::FixedWindow::MaximalEndComponentClassifier<ValueType>(this->mdp, unfoldedMECs, true, memoryStatesLabeling)
                    );
                } break;
            }
            storm::storage::BitVector goodStateSpace = mecClassifier->getGoodStateSpace();
            storm::storage::Scheduler<ValueType> scheduler = mecClassifier->getMaximalEndComponentScheduler();
            return storage::GoodStateSpaceAndScheduler<ValueType>(std::move(goodStateSpace), std::move(scheduler));
        }

        template <typename ValueType>
        storm::storage::BitVector BoundedWindowParityObjective<ValueType>::getGoodStateSpace() const {
            std::unique_ptr<sw::util::MaximalEndComponentClassifier<ValueType>> mecClassifier;
            switch (this->classificationMethod) {
                case MemorylessWindowGame: {
                    sw::storage::MaximalEndComponentDecompositionWindowParityGame<ValueType> mecGames(this->mdp, this->rewardModelName);
                    mecClassifier = std::unique_ptr<MaximalEndComponentClassifier<ValueType>>(
                            new MaximalEndComponentClassifier<ValueType>(this->mdp, mecGames)
                    );
                } break;
                default: {
                    sw::storage::MaximalEndComponentDecompositionUnfoldingParity<ValueType>
                    unfoldedMECs(this->mdp, this->rewardModelName, this->getUniformBound());
                    mecClassifier = std::unique_ptr<sw::FixedWindow::MaximalEndComponentClassifier<ValueType>>(
                            new sw::FixedWindow::MaximalEndComponentClassifier<ValueType>(this->mdp, unfoldedMECs)
                    );
                } break;
            }
            return std::move(mecClassifier->getGoodStateSpace());
        }

        template<typename ValueType>
        uint_fast64_t BoundedWindowParityObjective<ValueType>::getUniformBound() const {
            std::vector<ValueType> const& priorities = this->mdp.getRewardModel(this->rewardModelName).getStateRewardVector();
            ValueType d = -1 * storm::utility::infinity<ValueType>();
            storm::storage::MaximalEndComponentDecomposition<ValueType> mecDecomposition(this->mdp);
            for (storm::storage::MaximalEndComponent const& mec : mecDecomposition) {
                for (auto const& stateActionsPair : mec) {
                    uint_fast64_t const& state = stateActionsPair.first;
                    if (d < priorities[state]) {
                        d = priorities[state];
                    }
                }
            }
            auto d_int = storm::utility::convertNumber<uint_fast64_t>(storm::utility::ceil(d / 2));
            return d_int * this->mdp.getNumberOfStates();
        }

        template<typename ValueType>
        storage::GoodStateSpaceAndScheduler<ValueType>
        BoundedWindowParityObjective<ValueType>::produceGoodScheduler(bool memoryStatesLabeling) const {
            std::unique_ptr<sw::util::MaximalEndComponentClassifier<ValueType>> mecClassifier;
            switch (this->classificationMethod) {
                case MemorylessWindowGame: {
                    sw::storage::MaximalEndComponentDecompositionWindowParityGame<ValueType> mecGames(this->mdp, this->rewardModelName);
                    mecClassifier = std::unique_ptr<MaximalEndComponentClassifier<ValueType>>(
                            new MaximalEndComponentClassifier<ValueType>(this->mdp, mecGames, true)
                    );
                } break;
                default: {
                    sw::storage::MaximalEndComponentDecompositionUnfoldingParity<ValueType>
                    unfoldedMECs(this->mdp, this->rewardModelName, this->getUniformBound());
                    mecClassifier = std::unique_ptr<sw::FixedWindow::MaximalEndComponentClassifier<ValueType>>(
                            new sw::FixedWindow::MaximalEndComponentClassifier<ValueType>(this->mdp, unfoldedMECs, true, memoryStatesLabeling)
                    );
                } break;
            }
            storm::storage::BitVector goodStateSpace = mecClassifier->getGoodStateSpace();
            storm::storage::Scheduler<ValueType> scheduler = mecClassifier->getMaximalEndComponentScheduler();
            return storage::GoodStateSpaceAndScheduler<ValueType>(std::move(goodStateSpace), std::move(scheduler));
        }

        template<typename ValueType>
        sw::storage::ValuesAndScheduler<ValueType> performMaxProb(BoundedWindowObjective<ValueType> const& bwObjective, bool produceScheduler, bool memoryStatesLabeling) {
            if (produceScheduler) {
                storage::GoodStateSpaceAndScheduler<ValueType> goodStateSpaceAndScheduler = bwObjective.produceGoodScheduler(memoryStatesLabeling);
                std::unique_ptr<storm::storage::Scheduler<ValueType>>
                        scheduler = std::unique_ptr<storm::storage::Scheduler<ValueType>>(
                        new storm::storage::Scheduler<ValueType>(bwObjective.getMdp().getNumberOfStates(), goodStateSpaceAndScheduler.scheduler.getMemoryStructure())
                );
                storm::modelchecker::helper::MDPSparseModelCheckingHelperReturnType<ValueType>
                        result = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>().computeUntilProbabilities(
                        storm::Environment(),
                        storm::solver::OptimizationDirection::Maximize, // we maximize the probability of reaching good MECs
                        bwObjective.getMdp().getTransitionMatrix(),
                        bwObjective.getMdp().getBackwardTransitions(),
                        storm::storage::BitVector(bwObjective.getMdp().getNumberOfStates(), true),
                        goodStateSpaceAndScheduler.goodStateSpace,
                        false, // quantitative
                        true); // produce scheduler

                for (uint_fast64_t state = 0; state < bwObjective.getMdp().getNumberOfStates(); ++ state) {
                    for (uint_fast64_t memory = 0; memory < scheduler->getNumberOfMemoryStates(); ++ memory) {
                        if (goodStateSpaceAndScheduler.goodStateSpace[state]) {
                            scheduler->setChoice(goodStateSpaceAndScheduler.scheduler.getChoice(state, memory), state, memory);
                        } else {
                            scheduler->setChoice(result.scheduler->getChoice(state), state, memory);
                        }
                    }
                }

                return sw::storage::ValuesAndScheduler<ValueType>(std::move(result.values), std::move(scheduler));

            } else {
                storm::storage::BitVector goodStateSpace = bwObjective.getGoodStateSpace();
                storm::modelchecker::helper::MDPSparseModelCheckingHelperReturnType<ValueType>
                        result = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>().computeUntilProbabilities(
                        storm::Environment(),
                        storm::solver::OptimizationDirection::Maximize, // we maximize the probability of reaching good MECs
                        bwObjective.getMdp().getTransitionMatrix(),
                        bwObjective.getMdp().getBackwardTransitions(),
                        storm::storage::BitVector(bwObjective.getMdp().getNumberOfStates(), true),
                        goodStateSpace,
                        false, // quantitative
                        false); // do not produce scheduler

                return sw::storage::ValuesAndScheduler<ValueType>(std::move(result.values));
            }
        }

    template class BoundedWindowObjective<double>;
    template class BoundedWindowObjective<storm::RationalNumber>;
    template class BoundedWindowMeanPayoffObjective<double>;
    template class BoundedWindowMeanPayoffObjective<storm::RationalNumber>;
    template class BoundedWindowParityObjective<double>;

    template sw::storage::ValuesAndScheduler<double> performMaxProb<double>(BoundedWindowObjective<double> const& bwObjective, bool produceScheduler, bool memoryStatesLabeling);
    template sw::storage::ValuesAndScheduler<storm::RationalNumber> performMaxProb<storm::RationalNumber>(BoundedWindowObjective<storm::RationalNumber> const& bwObjective, bool produceScheduler, bool memoryStatesLabeling);

    }
}