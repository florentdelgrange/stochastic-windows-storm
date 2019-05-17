//
// Created by Florent Delgrange on 2019-02-01.
//

#include <stochastic-windows/fixedwindow/MaximalEndComponentClassifier.h>

namespace sw {
    namespace FixedWindow {

        template <typename ValueType>
        MaximalEndComponentClassifier<ValueType>::MaximalEndComponentClassifier(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                sw::storage::MaximalEndComponentDecompositionUnfolding<ValueType> const& mecDecompositionUnfolding,
                bool produceScheduler)
                : sw::util::MaximalEndComponentClassifier<ValueType>(mdp, mecDecompositionUnfolding, produceScheduler) {

            sw::DirectFixedWindow::WindowUnfolding<ValueType> const& windowUnfolding = mecDecompositionUnfolding.getUnfolding();
            storm::storage::BitVector allStates(windowUnfolding.getMatrix().getRowGroupCount(), true);
            // the state 0 of the unfolding of the EC is the sink state we want to avoid;
            storm::storage::BitVector sinkState(windowUnfolding.getMatrix().getRowGroupCount(), false);
            sinkState.set(0, true);
            // compute the set of safe states w.r.t. the sink state
            storm::storage::SparseMatrix<ValueType> unfoldingBackwardTransitions = windowUnfolding.getMatrix().transpose(true);
            storm::storage::BitVector unfoldingWinningSet = storm::utility::graph::performProb0E(
                    windowUnfolding.getMatrix(), windowUnfolding.getMatrix().getRowGroupIndices(),
                    unfoldingBackwardTransitions, allStates, sinkState);
            std::vector<boost::optional<storm::storage::BitVector>> mecStates(mecDecompositionUnfolding.size());
            for (uint_fast64_t k = 0; k < mecDecompositionUnfolding.size(); ++ k) {
                // We compute the set of states for which there exists a strategy satisfying a probability zero of reaching this sink state.
                // This set of states is the safe state space.
                mecStates[k] = storm::storage::BitVector(mdp.getNumberOfStates());
                // translate states belonging to the kth MEC to initial states belonging to the kth MEC in the unfolding
                storm::storage::BitVector initialCurrentMECStates(windowUnfolding.getMatrix().getRowGroupCount());
                for (uint_fast64_t const& state : mecDecompositionUnfolding[k].getStateSet()) {
                    mecStates[k]->set(state, true);
                    uint_fast64_t initialState = windowUnfolding.getInitialState(state);
                    if (unfoldingWinningSet[initialState]) {
                        this->safeStateSpace.set(state, true);
                    }
                    initialCurrentMECStates.set(initialState, true);
                }
                if (not (unfoldingWinningSet & initialCurrentMECStates).empty()) {
                    this->goodMECs.set(k, true);
                    this->goodStateSpace |= *mecStates[k];
                }
            }
            if (produceScheduler) {
                sw::DirectFixedWindow::WindowMemory<ValueType> windowMemory = windowUnfolding.generateMemory();
                this->mecScheduler = std::unique_ptr<storm::storage::Scheduler<ValueType>>(
                        new storm::storage::Scheduler<ValueType>(mdp.getNumberOfStates(), *windowMemory.memoryStructure)
                );
                storm::storage::SparseMatrix<ValueType> backwardTransitions = mdp.getTransitionMatrix().transpose(true);
                for (uint_fast64_t const& k : this->goodMECs) {
                    storm::utility::graph::computeSchedulerProb1E(*mecStates[k], // good states has a probability one of reaching safe states belonging to the same MEC
                                                                  mdp.getTransitionMatrix(), backwardTransitions, // transitions
                                                                  *mecStates[k], // phi
                                                                  *mecStates[k] & this->safeStateSpace, // psi
                                                                  *this->mecScheduler); // update scheduler
                }
                storm::storage::Scheduler<ValueType> unfoldingScheduler(windowUnfolding.getMatrix().getRowGroupCount());
                storm::utility::graph::computeSchedulerProb0E(unfoldingWinningSet, windowUnfolding.getMatrix(), unfoldingScheduler);
                std::vector<sw::DirectFixedWindow::StateValueWindowSize<ValueType>> unfoldingStatesMeaning = windowUnfolding.getNewStatesMeaning();
                for (uint_fast64_t unfoldingState = 1; unfoldingState < windowUnfolding.getMatrix().getRowGroupCount(); ++ unfoldingState) {
                    uint_fast64_t state = unfoldingStatesMeaning[unfoldingState].state;
                    uint_fast64_t memory = windowMemory.unfoldingToMemoryStatesMapping[unfoldingState];
                    this->mecScheduler->setChoice(unfoldingScheduler.getChoice(unfoldingState), state, memory);
                }
            }
        }

        template <typename ValueType>
        MaximalEndComponentClassifier<ValueType>::MaximalEndComponentClassifier(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                sw::storage::MaximalEndComponentDecompositionWindowGame<ValueType> const& mecDecompositionGame,
                bool produceScheduler)
                : sw::util::MaximalEndComponentClassifier<ValueType>(mdp, mecDecompositionGame, produceScheduler) {

            std::vector<std::unique_ptr<storm::storage::Scheduler<ValueType>>> schedulersVector(mecDecompositionGame.size());
            for (uint_fast64_t k = 0; k < mecDecompositionGame.size(); ++ k) {
                sw::game::WindowGame<ValueType> const& mecGame = mecDecompositionGame.getGame(k);
                sw::game::WinningSetAndScheduler<ValueType>
                gameResult = produceScheduler ? mecGame.produceSchedulerForDirectFW()
                                              : sw::game::WinningSetAndScheduler<ValueType>(mecGame.directFW());
                // If the winning set in the current MEC game is not empty, then the MEC is classified as good.
                if (not gameResult.winningSet.empty()) {
                    this->goodMECs.set(k, true);
                    // logical OR
                    this->goodStateSpace |= mecGame.getStateSpace();
                    this->safeStateSpace |= gameResult.winningSet;
                    schedulersVector[k] = std::move(gameResult.scheduler);
                }
            }
            if (produceScheduler) {
                // Memory transitions
                storm::storage::MemoryStructure::TransitionMatrix memoryUpdates(mecDecompositionGame.getMaximumWindowSize(), std::vector<boost::optional<storm::storage::BitVector>>(mecDecompositionGame.getMaximumWindowSize()));
                for (uint_fast64_t memory = 0; memory < mecDecompositionGame.getMaximumWindowSize(); ++ memory) {
                    for (uint_fast64_t next_memory = 1; next_memory < mecDecompositionGame.getMaximumWindowSize(); ++ next_memory) {
                        memoryUpdates[memory][next_memory] = storm::storage::BitVector(mdp.getNumberOfTransitions());
                        for (uint_fast64_t k = 0; k < mecDecompositionGame.size(); ++ k) {
                            if (schedulersVector[k]) {
                                *memoryUpdates[memory][next_memory] |= *schedulersVector[k]->getMemoryStructure()->getTransitionMatrix()[memory][next_memory];
                            }
                        }
                    }
                    memoryUpdates[memory][0] = storm::storage::BitVector(mdp.getNumberOfTransitions(), true);
                    for (uint_fast64_t k = 0; k < mecDecompositionGame.size(); ++ k) {
                        if (schedulersVector[k]) {
                            *memoryUpdates[memory][0] &= *schedulersVector[k]->getMemoryStructure()->getTransitionMatrix()[memory][0];
                        }
                    }
                }
                // state labels
                storm::models::sparse::StateLabeling stateLabeling(mecDecompositionGame.getMaximumWindowSize());
                // Initial memory states
                std::vector<uint_fast64_t> initialMemoryStates(mdp.getInitialStates().getNumberOfSetBits(), 0);
                auto initMemStateIt = initialMemoryStates.begin();
                for (auto const& initState : mdp.getInitialStates()) {
                    *initMemStateIt = 0;
                    ++ initMemStateIt;
                }
                storm::storage::MemoryStructure memoryStructure(std::move(memoryUpdates), std::move(stateLabeling), std::move(initialMemoryStates));
                this->mecScheduler = std::unique_ptr<storm::storage::Scheduler<ValueType>>(
                        new storm::storage::Scheduler<ValueType>(mdp.getNumberOfStates(), std::move(memoryStructure))
                );
                // reach the safe state space from the good state space
                storm::storage::SparseMatrix<ValueType> backwardTransitions = mdp.getTransitionMatrix().transpose(true);
                for (uint_fast64_t const& k : this->goodMECs) {
                    storm::utility::graph::computeSchedulerProb1E(mecDecompositionGame.getGame(k).getStateSpace(), // good states has a probability one of reaching safe states in the same MEC
                                                                  mdp.getTransitionMatrix(), backwardTransitions, // transitions
                                                                  mecDecompositionGame.getGame(k).getStateSpace(), // phi
                                                                  mecDecompositionGame.getGame(k).getStateSpace() & this->safeStateSpace, // psi
                                                                  *this->mecScheduler); // update scheduler
                }
                for (uint_fast64_t const& state : this->safeStateSpace) {
                    for (uint_fast64_t l = 0; l < mecDecompositionGame.getMaximumWindowSize(); ++ l) {
                        this->mecScheduler->setChoice(schedulersVector[mecDecompositionGame.getMecIndex(state)]->getChoice(state, l), state, l);
                    }
                }
            }
        }

        template class MaximalEndComponentClassifier<double>;
        template class MaximalEndComponentClassifier<storm::RationalNumber>;

    }
}