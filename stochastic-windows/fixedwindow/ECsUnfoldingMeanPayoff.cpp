//
// Created by Florent Delgrange on 2018-11-23.
//

#include "ECsUnfoldingMeanPayoff.h"

template<typename ValueType>
sw::FixedWindow::ECsUnfoldingMeanPayoff<ValueType>::ECsUnfoldingMeanPayoff(
        storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>>& mdp,
        std::string const& rewardModelName,
        uint_fast64_t const& l_max)
        : mecDecomposition(mdp){

    this->mecIndices = std::vector<uint_fast64_t>(mdp.getNumberOfStates());
    uint_fast64_t k = 0;
    for (storm::storage::MaximalEndComponent const &mec: this->mecDecomposition){
        ++ k;
        storm::storage::BitVector initialStates(mdp.getNumberOfStates(), false);
        std::vector<std::vector<uint_fast64_t>> enabledActions(mdp.getNumberOfStates());

        for (auto state: mec.getStateSet()){
            initialStates.set(state, true);
            for (auto action: mec.getChoicesForState(state)) {
                enabledActions[state].push_back(action);
            }
            this->mecIndices[state] = k;
        }

        this->unfoldedECs.push_back(
                sw::DirectFixedWindow::UnfoldingMeanPayoff<ValueType>(
                        mdp, rewardModelName, l_max, initialStates, enabledActions
                        ));
    }
}

template <typename ValueType>
storm::storage::MaximalEndComponentDecomposition<ValueType>&
sw::FixedWindow::ECsUnfoldingMeanPayoff<ValueType>::getMaximalEndComponentDecomposition() {
    return this->mecDecomposition;
}

template <typename ValueType>
uint_fast64_t sw::FixedWindow::ECsUnfoldingMeanPayoff<ValueType>::getMecIndex(uint_fast64_t state) {
    return mecIndices[state];
}

template <typename ValueType>
std::pair<uint_fast64_t, uint_fast64_t> sw::FixedWindow::ECsUnfoldingMeanPayoff<ValueType>::getNewIndex(uint_fast64_t state,
                                                                                           ValueType currentSumOfWeights,
                                                                                           uint_fast64_t currentWindowSize) {
    uint_fast64_t k = getMecIndex(state);
    if ( !k ) return std::make_pair(0, 0);
    else return std::make_pair(k, this->unfoldedECs[k - 1].getNewIndex(state, currentSumOfWeights, currentWindowSize));
}

template <typename ValueType>
storm::storage::SparseMatrix<ValueType>& sw::FixedWindow::ECsUnfoldingMeanPayoff<ValueType>::getUnfoldedMatrix(uint_fast64_t mec) {
    return this->unfoldedECs[mec - 1].getMatrix();
}

template<typename ValueType>
uint_fast64_t sw::FixedWindow::ECsUnfoldingMeanPayoff<ValueType>::getNumberOfUnfoldedECs() {
    return this->unfoldedECs.size();
}

template<typename ValueType>
std::vector<sw::DirectFixedWindow::StateValueWindowSize<ValueType>>
sw::FixedWindow::ECsUnfoldingMeanPayoff<ValueType>::getNewStatesMeaning(uint_fast64_t k) {
    return this->unfoldedECs[k - 1].getNewStatesMeaning();
}

template<typename ValueType>
void sw::FixedWindow::ECsUnfoldingMeanPayoff<ValueType>::printToStream(std::ostream &out, uint_fast64_t k) {
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
std::shared_ptr<storm::models::sparse::Mdp<ValueType>> sw::FixedWindow::ECsUnfoldingMeanPayoff<ValueType>::unfoldingAsMDP(uint_fast64_t k) {
    storm::storage::SparseMatrix<ValueType> *unfoldedECMatrix = &(this->getUnfoldedMatrix(k));
    storm::storage::sparse::ModelComponents<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> modelComponents(*unfoldedECMatrix, storm::models::sparse::StateLabeling(unfoldedECMatrix->getColumnCount()));
    std::shared_ptr<storm::models::sparse::Model<ValueType>> unfoldedECModel =
            storm::utility::builder::buildModelFromComponents(storm::models::ModelType::Mdp, std::move(modelComponents));
    std::shared_ptr<storm::models::sparse::Mdp<ValueType>> new_mdp = unfoldedECModel->template as<storm::models::sparse::Mdp<ValueType>>();
    return new_mdp;
}

template class sw::FixedWindow::ECsUnfoldingMeanPayoff<double>;
template class sw::FixedWindow::ECsUnfoldingMeanPayoff<storm::RationalNumber>;
