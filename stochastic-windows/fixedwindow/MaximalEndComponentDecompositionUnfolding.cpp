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
          mecIndices(mdp.getNumberOfStates()) {
    this->unfoldedECs.reserve(this->size());
}

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

template <typename ValueType>
void sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::performMECsUnfolding(
        storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
        std::string const &rewardModelName, uint_fast64_t const &l_max) {

    uint_fast64_t k = 0;
    for (storm::storage::MaximalEndComponent const &mec: *this){
        ++ k;
        storm::storage::BitVector initialStates(mdp.getNumberOfStates(), false);
        storm::storage::BitVector enabledActions(mdp.getNumberOfChoices(), false);

        for (auto state: mec.getStateSet()){
            initialStates.set(state, true);
            for (auto action: mec.getChoicesForState(state)) {
                enabledActions.set(action, true);
            }
            this->mecIndices[state] = k;
        }
        unfoldEC(mdp, rewardModelName, l_max, initialStates, enabledActions);
    }
}

template <typename ValueType>
void sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<ValueType>::unfoldEC(
        storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
        std::string const &rewardModelName, uint_fast64_t const &l_max, storm::storage::BitVector const &initialStates,
        storm::storage::BitVector const& enabledActions) {
    std::unique_ptr<sw::DirectFixedWindow::WindowUnfolding<ValueType>> window_ptr(
            new sw::DirectFixedWindow::WindowUnfoldingMeanPayoff<ValueType>(
                    mdp, rewardModelName, l_max, initialStates, enabledActions));
    this->unfoldedECs.push_back(std::move(window_ptr));
}

template <typename ValueType>
void sw::storage::MaximalEndComponentDecompositionUnfoldingParity<ValueType>::unfoldEC(
        storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
        std::string const &rewardModelName, uint_fast64_t const &l_max, storm::storage::BitVector const &initialStates,
        storm::storage::BitVector const& enabledActions) {
    std::unique_ptr<sw::DirectFixedWindow::WindowUnfolding<ValueType>> window_ptr(
            new sw::DirectFixedWindow::WindowUnfoldingParity<ValueType>(
                    mdp, rewardModelName, l_max, initialStates, enabledActions));
    this->unfoldedECs.push_back(std::move(window_ptr));
}

template <typename ValueType>
uint_fast64_t sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::getMecIndex(uint_fast64_t state) {
    STORM_LOG_ASSERT(mecIndices[state] == 0, "The state" << state << "does not belong to any MEC.");
    return mecIndices[state] - 1;
}

template <typename ValueType>
std::pair<uint_fast64_t, uint_fast64_t> sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::getNewIndex(uint_fast64_t state,
        ValueType currentSumOfWeights,
        uint_fast64_t currentWindowSize) {
    uint_fast64_t k = mecIndices[state];
    if ( !k ) return std::make_pair(0, 0);
    else return std::make_pair(k - 1, this->unfoldedECs[k - 1]->getNewIndex(state, currentSumOfWeights, currentWindowSize));
}

template <typename ValueType>
storm::storage::SparseMatrix<ValueType> const& sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::getUnfoldedMatrix(uint_fast64_t mec) {
    return this->unfoldedECs[mec]->getMatrix();
}

template <typename ValueType>
storm::storage::SparseMatrix<ValueType> const& sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::getUnfoldedMatrix(uint_fast64_t mec) const {
    return this->unfoldedECs[mec]->getMatrix();
}

template<typename ValueType>
std::vector<sw::DirectFixedWindow::StateValueWindowSize<ValueType>>
sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::getNewStatesMeaning(uint_fast64_t k) {
    return this->unfoldedECs[k]->getNewStatesMeaning();
}

template<typename ValueType>
uint_fast64_t sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::getInitialState(uint_fast64_t k, uint_fast64_t initialState) {
    if (this->mecIndices[initialState] != 0) {
        return this->unfoldedECs[k]->getInitialState(initialState);
    } else {
        return 0;
    }
}

template<typename ValueType>
uint_fast64_t sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::getInitialState(uint_fast64_t k, uint_fast64_t initialState) const {
    if (this->mecIndices[initialState] != 0) {
        return this->unfoldedECs[k]->getInitialState(initialState);
    } else {
        return 0;
    }
}

template<typename ValueType>
void sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::printToStream(std::ostream &out, uint_fast64_t k) {
    out << this->getUnfoldedMatrix(k) << "\n";
    out << "where" << "\n";
    std::vector<sw::DirectFixedWindow::StateValueWindowSize<ValueType>>
            newStatesMeaning = this->getNewStatesMeaning(k);
    out << "state 0 is ⊥" << "\n";
    for(uint_fast64_t state = 1; state < this->getUnfoldedMatrix(k).getRowGroupCount(); ++ state) {
        std::ostringstream stream;
        stream << "(s" << newStatesMeaning[state].state << ", " <<
               newStatesMeaning[state].currentValue << ", " <<
               newStatesMeaning[state].currentWindowSize << ")";
        out << "state " << state << " is " << stream.str() << "\n";
    }
}

template<typename ValueType>
std::shared_ptr<storm::models::sparse::Mdp<ValueType>> sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType>::unfoldingAsMDP(uint_fast64_t k) {
    storm::storage::SparseMatrix<ValueType> const& unfoldedECMatrix = this->getUnfoldedMatrix(k);
    storm::storage::sparse::ModelComponents<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> modelComponents(unfoldedECMatrix, storm::models::sparse::StateLabeling(unfoldedECMatrix.getColumnCount()));
    std::shared_ptr<storm::models::sparse::Model<ValueType>> unfoldedECModel =
            storm::utility::builder::buildModelFromComponents(storm::models::ModelType::Mdp, std::move(modelComponents));
    std::shared_ptr<storm::models::sparse::Mdp<ValueType>> new_mdp = unfoldedECModel->template as<storm::models::sparse::Mdp<ValueType>>();
    return new_mdp;
}

template class sw::storage::MaximalEndComponentDecompositionUnfolding<double>;
template class sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<double>;
template class sw::storage::MaximalEndComponentDecompositionUnfoldingParity<double>;
template class sw::storage::MaximalEndComponentDecompositionUnfolding<storm::RationalNumber>;
template class sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<storm::RationalNumber>;
