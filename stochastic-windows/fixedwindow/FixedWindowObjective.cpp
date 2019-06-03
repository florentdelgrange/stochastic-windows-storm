//
// Created by Florent Delgrange on 2019-02-05.
//

#include "FixedWindowObjective.h"

namespace sw {
    namespace FixedWindow {

        template <typename ValueType>
        FixedWindowObjective<ValueType>::FixedWindowObjective(
                const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                std::string const &rewardModelName, uint_fast64_t const &l_max):
                WindowObjective<ValueType>(mdp, rewardModelName),
                l_max(l_max) {}

        template <typename ValueType>
        uint_fast64_t FixedWindowObjective<ValueType>::getMaximumWindowSize() const {
            return this->l_max;
        }

        template <typename ValueType>
        FixedWindowMeanPayoffObjective<ValueType>::FixedWindowMeanPayoffObjective(
                const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                std::string const &rewardModelName, uint_fast64_t const &l_max, bool windowGameBasedClassification)
                : FixedWindowObjective<ValueType>(mdp, rewardModelName, l_max),
                  windowGameClassification(windowGameBasedClassification) {}

        template <typename ValueType>
        FixedWindowParityObjective<ValueType>::FixedWindowParityObjective(
                const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                std::string const &rewardModelName, uint_fast64_t const &l_max)
                : FixedWindowObjective<ValueType>(mdp, rewardModelName, l_max) {}

        template <typename ValueType>
        storm::storage::BitVector FixedWindowMeanPayoffObjective<ValueType>::getGoodStateSpace() const {
            std::unique_ptr<MaximalEndComponentClassifier<ValueType>> mecClassifier;
            if (this->windowGameClassification) {
                sw::storage::MaximalEndComponentDecompositionWindowMeanPayoffGame<ValueType>
                mecGames(this->mdp, this->rewardModelName, this->l_max);
                mecClassifier = std::unique_ptr<MaximalEndComponentClassifier<ValueType>>(
                        new MaximalEndComponentClassifier<ValueType>(this->mdp, mecGames));
            } else {
                sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<ValueType>
                unfoldedMECs(this->mdp, this->rewardModelName, this->l_max);
                mecClassifier = std::unique_ptr<MaximalEndComponentClassifier<ValueType>>(
                        new MaximalEndComponentClassifier<ValueType>(this->mdp, unfoldedMECs));
            }
            return std::move(mecClassifier->getGoodStateSpace());
        }

        template<typename ValueType>
        storage::GoodStateSpaceAndScheduler<ValueType>
        FixedWindowMeanPayoffObjective<ValueType>::produceGoodScheduler(bool memoryStatesLabeling) const {
            std::unique_ptr<MaximalEndComponentClassifier<ValueType>> mecClassifier;
            if (this->windowGameClassification) {
                sw::storage::MaximalEndComponentDecompositionWindowMeanPayoffGame<ValueType>
                mecGames(this->mdp, this->rewardModelName, this->l_max);
                mecClassifier = std::unique_ptr<MaximalEndComponentClassifier<ValueType>>(
                        new MaximalEndComponentClassifier<ValueType>(this->mdp, mecGames, true, memoryStatesLabeling));
            } else {
                sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<ValueType>
                unfoldedMECs(this->mdp, this->rewardModelName, this->l_max);
                mecClassifier = std::unique_ptr<MaximalEndComponentClassifier<ValueType>>(
                        new MaximalEndComponentClassifier<ValueType>(this->mdp, unfoldedMECs, true, memoryStatesLabeling));
            }
            storm::storage::BitVector goodStateSpace = mecClassifier->getGoodStateSpace();
            storm::storage::Scheduler<ValueType> scheduler = mecClassifier->getMaximalEndComponentScheduler();
            return storage::GoodStateSpaceAndScheduler<ValueType>(std::move(goodStateSpace), std::move(scheduler));
        }

        template <typename ValueType>
        storm::storage::BitVector FixedWindowParityObjective<ValueType>::getGoodStateSpace() const {
            sw::storage::MaximalEndComponentDecompositionUnfoldingParity<ValueType>
            unfoldedMECs(this->mdp, this->rewardModelName, this->l_max);
            MaximalEndComponentClassifier<ValueType> mecClassifier(this->mdp, unfoldedMECs);
            return std::move(mecClassifier.getGoodStateSpace());
        }

        template<typename ValueType>
        storage::GoodStateSpaceAndScheduler<ValueType>
        FixedWindowParityObjective<ValueType>::produceGoodScheduler(bool memoryStatesLabeling) const {
            sw::storage::MaximalEndComponentDecompositionUnfoldingParity<ValueType>
            unfoldedMECs(this->mdp, this->rewardModelName, this->l_max);
            MaximalEndComponentClassifier<ValueType> mecClassifier(this->mdp, unfoldedMECs, true, memoryStatesLabeling);
            storm::storage::BitVector goodStateSpace = mecClassifier.getGoodStateSpace();
            storm::storage::Scheduler<ValueType> scheduler = mecClassifier.getMaximalEndComponentScheduler();
            return storage::GoodStateSpaceAndScheduler<ValueType>(std::move(goodStateSpace), std::move(scheduler));
        }

        template<typename ValueType>
        sw::storage::ValuesAndScheduler<ValueType> performMaxProb(FixedWindowObjective<ValueType> const& fwObjective, bool produceScheduler, bool memoryStatesLabeling) {
            if (produceScheduler) {
                storage::GoodStateSpaceAndScheduler<ValueType> goodStateSpaceAndScheduler = fwObjective.produceGoodScheduler(memoryStatesLabeling);
                std::unique_ptr<storm::storage::Scheduler<ValueType>>
                scheduler = std::unique_ptr<storm::storage::Scheduler<ValueType>>(
                        new storm::storage::Scheduler<ValueType>(fwObjective.getMdp().getNumberOfStates(), goodStateSpaceAndScheduler.scheduler.getMemoryStructure())
                );
                storm::modelchecker::helper::MDPSparseModelCheckingHelperReturnType<ValueType>
                result = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>().computeUntilProbabilities(
                        storm::Environment(),
                        storm::solver::OptimizationDirection::Maximize, // we want to maximize the probability of reaching good states
                        fwObjective.getMdp().getTransitionMatrix(),
                        fwObjective.getMdp().getBackwardTransitions(),
                        storm::storage::BitVector(fwObjective.getMdp().getNumberOfStates(), true),
                        goodStateSpaceAndScheduler.goodStateSpace,
                        false, // quantitative
                        true); // produce scheduler

                for (uint_fast64_t state = 0; state < fwObjective.getMdp().getNumberOfStates(); ++ state) {
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
                storm::storage::BitVector goodStateSpace = fwObjective.getGoodStateSpace();
                storm::modelchecker::helper::MDPSparseModelCheckingHelperReturnType<ValueType>
                result = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>().computeUntilProbabilities(
                        storm::Environment(),
                        storm::solver::OptimizationDirection::Maximize, // we maximize the probability of reaching good MECs
                        fwObjective.getMdp().getTransitionMatrix(),
                        fwObjective.getMdp().getBackwardTransitions(),
                        storm::storage::BitVector(fwObjective.getMdp().getNumberOfStates(), true),
                        goodStateSpace,
                        false, // quantitative
                        false); // do not produce scheduler

                return sw::storage::ValuesAndScheduler<ValueType>(std::move(result.values));
            }
        }

        template class FixedWindowObjective<double>;
        template class FixedWindowObjective<storm::RationalNumber>;
        template class FixedWindowMeanPayoffObjective<double>;
        template class FixedWindowMeanPayoffObjective<storm::RationalNumber>;
        template class FixedWindowParityObjective<double>;

        template sw::storage::ValuesAndScheduler<double> performMaxProb<double>(FixedWindowObjective<double> const& fwObjective, bool produceScheduler, bool memoryStatesLabeling);
        template sw::storage::ValuesAndScheduler<storm::RationalNumber> performMaxProb<storm::RationalNumber>(FixedWindowObjective<storm::RationalNumber> const& fwObjective, bool produceScheduler, bool memoryStatesLabeling);

    }
}