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
            }
            else {
                sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<ValueType>
                unfoldedMECs(this->mdp, this->rewardModelName, this->l_max);
                mecClassifier = std::unique_ptr<MaximalEndComponentClassifier<ValueType>>(
                        new MaximalEndComponentClassifier<ValueType>(this->mdp, unfoldedMECs));
            }
            return std::move(mecClassifier->getGoodStateSpace());
        }

        template <typename ValueType>
        storm::storage::BitVector FixedWindowParityObjective<ValueType>::getGoodStateSpace() const {
            sw::storage::MaximalEndComponentDecompositionUnfoldingParity<ValueType>
            unfoldedMECs(this->mdp, this->rewardModelName, this->l_max);
            MaximalEndComponentClassifier<ValueType> mecClassifier(this->mdp, unfoldedMECs);
            return std::move(mecClassifier.getGoodStateSpace());
        }

        template<typename ValueType>
        std::vector<ValueType> performMaxProb(FixedWindowObjective<ValueType> const& fwObjective) {
            return storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>().computeUntilProbabilities(
                storm::Environment(),
                storm::solver::SolveGoal<ValueType>(false), // we want to maximize the probability of reaching good states
                fwObjective.getMdp().getTransitionMatrix(),
                fwObjective.getMdp().getTransitionMatrix().transpose(true),
                storm::storage::BitVector(fwObjective.getMdp().getNumberOfStates(), true),
                fwObjective.getGoodStateSpace(),
                false,
                false).values;
        }

        template<typename ValueType>
        ValueType performMaxProb(uint_fast64_t state, FixedWindowObjective<ValueType> const& fwObjective) {
            return performMaxProb(fwObjective)[state];
        }

        template class FixedWindowObjective<double>;
        template class FixedWindowObjective<storm::RationalNumber>;
        template class FixedWindowMeanPayoffObjective<double>;
        template class FixedWindowMeanPayoffObjective<storm::RationalNumber>;
        template class FixedWindowParityObjective<double>;

        template std::vector<double> performMaxProb<double>(FixedWindowObjective<double> const& fwObjective);
        template std::vector<storm::RationalNumber> performMaxProb<storm::RationalNumber>(FixedWindowObjective<storm::RationalNumber> const& fwObjective);
        template double performMaxProb<double>(uint_fast64_t state, FixedWindowObjective<double> const& fwObjective);
        template storm::RationalNumber performMaxProb<storm::RationalNumber>(uint_fast64_t state, FixedWindowObjective<storm::RationalNumber> const& fwObjective);

    }
}