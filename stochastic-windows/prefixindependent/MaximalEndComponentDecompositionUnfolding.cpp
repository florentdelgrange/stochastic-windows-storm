//
// Created by Florent Delgrange on 2018-11-23.
//

#include "MaximalEndComponentDecompositionUnfolding.h"

template<typename ValueType>
sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::MaximalEndComponentDecompositionUnfolding(
        storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
        std::string const &rewardModelName,
        uint_fast64_t const &l_max)
        : storm::storage::MaximalEndComponentDecomposition<ValueType>(mdp),
          l_max(l_max),
          mecIndices(mdp.getNumberOfStates()) {}

template<typename ValueType>
sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<ValueType>::MaximalEndComponentDecompositionUnfoldingMeanPayoff(
        storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
        std::string const& rewardModelName,
        uint_fast64_t const& l_max)
        : MaximalEndComponentDecompositionUnfolding<ValueType>(mdp, rewardModelName, l_max) {
    MaximalEndComponentDecompositionUnfolding<ValueType>::performMECsUnfolding(mdp, rewardModelName, l_max);
}

template<typename ValueType>
sw::storage::MaximalEndComponentDecompositionUnfoldingParity<ValueType>::MaximalEndComponentDecompositionUnfoldingParity(
        storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
        std::string const& rewardModelName,
        uint_fast64_t const& l_max)
        : MaximalEndComponentDecompositionUnfolding<ValueType>(mdp, rewardModelName, l_max) {
    MaximalEndComponentDecompositionUnfolding<ValueType>::performMECsUnfolding(mdp, rewardModelName, l_max);
}

template<typename ValueType>
uint_fast64_t sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::getMaximumWindowSize() const {
    return this->l_max;
}

template <typename ValueType>
void sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::performMECsUnfolding(
        storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
        std::string const &rewardModelName, uint_fast64_t const &l_max) {

    uint_fast64_t k = 0;
    storm::storage::BitVector initialStates(mdp.getNumberOfStates(), false);
    storm::storage::BitVector enabledActions(mdp.getNumberOfChoices(), false);

    for (storm::storage::MaximalEndComponent const &mec: *this){
        ++ k;

        for (auto state: mec.getStateSet()){
            initialStates.set(state, true);
            for (auto action: mec.getChoicesForState(state)) {
                enabledActions.set(action, true);
            }
            this->mecIndices[state] = k;
        }
    }
    unfold(mdp, rewardModelName, l_max, initialStates, enabledActions);
}

template <typename ValueType>
void sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<ValueType>::unfold(
        storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
        std::string const &rewardModelName, uint_fast64_t const &l_max, storm::storage::BitVector const &initialStates,
        storm::storage::BitVector const &enabledActions) {
    this->unfoldedECs = std::unique_ptr<sw::DirectFixedWindow::WindowUnfolding<ValueType>>(
            new sw::DirectFixedWindow::WindowUnfoldingMeanPayoff<ValueType>(
                    mdp, rewardModelName, l_max, initialStates, enabledActions)
            );
}

template <typename ValueType>
void sw::storage::MaximalEndComponentDecompositionUnfoldingParity<ValueType>::unfold(
        storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
        std::string const &rewardModelName, uint_fast64_t const &l_max, storm::storage::BitVector const &initialStates,
        storm::storage::BitVector const &enabledActions) {
    this->unfoldedECs = std::unique_ptr<sw::DirectFixedWindow::WindowUnfolding<ValueType>>(
            new sw::DirectFixedWindow::WindowUnfoldingParity<ValueType>(
                    mdp, rewardModelName, l_max, initialStates, enabledActions)
            );
}

template <typename ValueType>
uint_fast64_t sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::getMecIndex(uint_fast64_t state) {
    STORM_LOG_ASSERT(mecIndices[state] == 0, "The state " << state << " does not belong to any MEC.");
    return mecIndices[state] - 1;
}

template<typename ValueType>
sw::DirectFixedWindow::WindowUnfolding<ValueType> const&
sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::getUnfolding() const {
    return *this->unfoldedECs;
}


template class sw::storage::MaximalEndComponentDecompositionUnfolding<double>;
template class sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<double>;
template class sw::storage::MaximalEndComponentDecompositionUnfoldingParity<double>;
template class sw::storage::MaximalEndComponentDecompositionUnfolding<storm::RationalNumber>;
template class sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<storm::RationalNumber>;
