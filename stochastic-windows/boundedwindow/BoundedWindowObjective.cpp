//
// Created by Florent Delgrange on 2019-03-26.
//

#include "BoundedWindowObjective.h"

namespace sw {
    namespace BoundedWindow {

        template <typename ValueType>
        BoundedWindowObjective<ValueType>::BoundedWindowObjective(
                const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                std::string const &rewardModelName)
                : WindowObjective<ValueType>(mdp, rewardModelName){}

        template <typename ValueType>
        BoundedWindowMeanPayoffObjective<ValueType>::BoundedWindowMeanPayoffObjective(
                const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                std::string const &rewardModelName)
                : BoundedWindowObjective<ValueType>(mdp, rewardModelName) {}

        template <typename ValueType>
        BoundedWindowParityObjective<ValueType>::BoundedWindowParityObjective(
                const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                std::string const &rewardModelName)
                : BoundedWindowObjective<ValueType>(mdp, rewardModelName) {}

        template <typename ValueType>
        storm::storage::BitVector BoundedWindowMeanPayoffObjective<ValueType>::getGoodStateSpace() const {
            std::unique_ptr<MaximalEndComponentClassifier<ValueType>> mecClassifier;
            sw::storage::MaximalEndComponentDecompositionWindowMeanPayoffGame<ValueType>
            mecGames(this->mdp, this->rewardModelName);
            mecClassifier = std::unique_ptr<MaximalEndComponentClassifier<ValueType>>(
                    new MaximalEndComponentClassifier<ValueType>(this->mdp, mecGames)
            );
            return std::move(mecClassifier->getGoodStateSpace());
        }

        template <typename ValueType>
        storm::storage::BitVector BoundedWindowParityObjective<ValueType>::getGoodStateSpace() const {
            std::unique_ptr<MaximalEndComponentClassifier<ValueType>> mecClassifier;
            sw::storage::MaximalEndComponentDecompositionWindowParityGame<ValueType>
            mecGames(this->mdp, this->rewardModelName);
            mecClassifier = std::unique_ptr<MaximalEndComponentClassifier<ValueType>>(
                    new MaximalEndComponentClassifier<ValueType>(this->mdp, mecGames)
            );
            return std::move(mecClassifier->getGoodStateSpace());
        }

        template<typename ValueType>
        std::vector<ValueType> performMaxProb(BoundedWindowObjective<ValueType> const& bwObjective) {
        return storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>().computeUntilProbabilities(
                storm::Environment(),
                storm::solver::SolveGoal<ValueType>(false), // we want to maximize the probability of reaching good states
                bwObjective.getMdp().getTransitionMatrix(),
                bwObjective.getMdp().getTransitionMatrix().transpose(true),
        storm::storage::BitVector(bwObjective.getMdp().getNumberOfStates(), true),
        bwObjective.getGoodStateSpace(),
        false,
        false).values;
    }

    template<typename ValueType>
    ValueType performMaxProb(uint_fast64_t state, BoundedWindowObjective<ValueType> const& bwObjective) {
        return performMaxProb(bwObjective)[state];
    }

    template class BoundedWindowObjective<double>;
    template class BoundedWindowObjective<storm::RationalNumber>;
    template class BoundedWindowMeanPayoffObjective<double>;
    template class BoundedWindowMeanPayoffObjective<storm::RationalNumber>;
    template class BoundedWindowParityObjective<double>;

    template std::vector<double> performMaxProb<double>(BoundedWindowObjective<double> const& bwObjective);
    template std::vector<storm::RationalNumber> performMaxProb<storm::RationalNumber>(BoundedWindowObjective<storm::RationalNumber> const& bwObjective);
    template double performMaxProb<double>(uint_fast64_t state, BoundedWindowObjective<double> const& bwObjective);
    template storm::RationalNumber performMaxProb<storm::RationalNumber>(uint_fast64_t state, BoundedWindowObjective<storm::RationalNumber> const& bwObjective);

    }
}