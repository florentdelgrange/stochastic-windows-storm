//
// Created by Florent Delgrange on 2018-11-28.
//

#include <stochastic-windows/fixedwindow/MeanPayoff.h>

namespace sw {
    namespace FixedWindow {

        template <typename ValueType>
        MeanPayoff<ValueType>::MeanPayoff(storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                                            std::string const& rewardModelName,
                                            uint_fast64_t const& l_max)
                                            : unfoldedECs(mdp, rewardModelName, l_max),
                                              goodECs(unfoldedECs.size(), false),
                                              safeStates(mdp.getNumberOfStates(), false),
                                              unfoldingWinningSet(unfoldedECs.size()) {
            this->l_max = l_max;
            std::vector<uint_fast64_t> initialStates(mdp.getNumberOfStates());
            // std::vector<storm::storage::Scheduler<ValueType>> schedulersForUnfoldedECs;

            // For each unfolded EC, we start by computing the safe states w.r.t. the sink state
            for (uint_fast64_t k = 0; k < unfoldedECs.size(); ++ k) {
                storm::storage::SparseMatrix<ValueType> const& unfoldedECMatrix = this->unfoldedECs.getUnfoldedMatrix(k);
                storm::storage::BitVector allStates(unfoldedECMatrix.getRowGroupCount(), true);
                // the state 0 of the unfolding of the EC is the sink state we want to avoid;
                storm::storage::BitVector sinkState(unfoldedECMatrix.getRowGroupCount(), false);
                sinkState.set(0, true);

                // we compute the set of states for which there exists a strategy satisfying a a probability zero of reaching this sink state.
                // If it is the case for some state, then this strategy allows to always avoid the sink state.
                this->unfoldingWinningSet[k] = storm::utility::graph::performProb0E(unfoldedECMatrix,
                                                                                    unfoldedECMatrix.getRowGroupIndices(),
                                                                                    unfoldedECMatrix.transpose(true),
                                                                                    allStates,
                                                                                    sinkState);
                storm::storage::BitVector winningSet(mdp.getNumberOfStates(), false);
                for (uint_fast64_t state : unfoldedECs[k].getStateSet()) {
                    initialStates[state] = unfoldedECs.getInitialState(k, state);
                    if (this->unfoldingWinningSet[k][initialStates[state]]) {
                        winningSet.set(state, true);
                    }
                }
                std::cout << "safe states of the unfolding " << k << "=" << this->unfoldingWinningSet[k] << std::endl;
                std::cout << "related winning set in the original MDP =" << winningSet << std::endl;
                // if there exists at least one good state, then as we are in an EC we know that there exists a strategy
                // allowing to reach it with probability one. Once this state is reached, we can always avoid the sink
                // state by applying this strategy.
                if (not this->unfoldingWinningSet[k].empty()){
                    this->goodECs.set(k, true);
                }
                for (auto const unfoldingState : this->unfoldingWinningSet[k]) {
                    uint_fast64_t state = unfoldedECs.getNewStatesMeaning(k)[unfoldingState].state;
                    this->safeStates.set(state, true);
                }
                /* scheduler support
                schedulersForUnfoldedECs.push_back(storm::storage::Scheduler<ValueType>(unfoldedECMatrix->getRowGroupCount()));
                storm::utility::graph::computeSchedulerProb0E(safeStates, *unfoldedECMatrix, schedulersForUnfoldedECs[k - 1]);
                schedulersForUnfoldedECs[k - 1].printToStream(std::cout);
                */
            }
            std::cout << "safe states of the unfolding=" << this->safeStates << std::endl;
        }

        template class MeanPayoff<double>;
        template class MeanPayoff<storm::RationalNumber>;
    }
}