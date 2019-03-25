//
// Created by Florent Delgrange on 2019-02-01.
//

#include <stochastic-windows/fixedwindow/MaximalEndComponentClassifier.h>

namespace sw {
    namespace FixedWindow {

        template <typename ValueType>
        MaximalEndComponentClassifier<ValueType>::MaximalEndComponentClassifier(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType> const& mecDecompositionUnfolding)
                : sw::util::MaximalEndComponentClassifier<ValueType>(mdp, mecDecompositionUnfolding) {

            // For each unfolded EC, we start by computing the safe states w.r.t. the sink state
            for (uint_fast64_t k = 0; k < mecDecompositionUnfolding.size(); ++ k) {
                storm::storage::SparseMatrix<ValueType> const& unfoldedECMatrix = mecDecompositionUnfolding.getUnfoldedMatrix(k);
                storm::storage::BitVector allStates(unfoldedECMatrix.getRowGroupCount(), true);
                // the state 0 of the unfolding of the EC is the sink state we want to avoid;
                storm::storage::BitVector sinkState(unfoldedECMatrix.getRowGroupCount(), false);
                sinkState.set(0, true);
                // we compute the set of states for which there exists a strategy satisfying a a probability zero of reaching this sink state.
                // If it is the case for some state, then this strategy allows to always avoid the sink state.
                storm::storage::BitVector unfoldingWinningSet = storm::utility::graph::performProb0E(
                        unfoldedECMatrix,unfoldedECMatrix.getRowGroupIndices(), unfoldedECMatrix.transpose(true), allStates, sinkState);
                // If the winning set in the unfolding of the current MEC is not empty, then the MEC is classified as good.
                if (not unfoldingWinningSet.empty()) {
                    this->goodMECs.set(k, true);
                    for (uint_fast64_t state : mecDecompositionUnfolding[k].getStateSet()) {
                        this->goodStateSpace.set(state, true);
                        uint_fast64_t initialState = mecDecompositionUnfolding.getInitialState(k, state);
                        if (unfoldingWinningSet[initialState]) {
                            this->safeStateSpace.set(state, true);
                        }
                    }
                }
            }
        }

        template <typename ValueType>
        MaximalEndComponentClassifier<ValueType>::MaximalEndComponentClassifier(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                sw::storage::MaximalEndComponentDecompositionWindowGame<ValueType> const& mecDecompositionGame)
                : sw::util::MaximalEndComponentClassifier<ValueType>(mdp, mecDecompositionGame) {

            for (uint_fast64_t k = 0; k < mecDecompositionGame.size(); ++ k) {
                sw::game::WindowGame<ValueType> const& mecGame = mecDecompositionGame.getGame(k);
                storm::storage::BitVector winningSet = mecGame.directFW();
                // If the winning set in the current MEC game is not empty, then the MEC is classified as good.
                if (not winningSet.empty()) {
                    this->goodMECs.set(k, true);
                    // logical OR
                    this->goodStateSpace |= mecGame.getStateSpace();
                    this->safeStateSpace |= winningSet;
                }
            }
        }

        template class MaximalEndComponentClassifier<double>;
        template class MaximalEndComponentClassifier<storm::RationalNumber>;

    }
}