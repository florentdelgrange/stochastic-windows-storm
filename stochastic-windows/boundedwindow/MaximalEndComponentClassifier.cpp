//
// Created by Florent Delgrange on 2019-02-01.
//

#include <stochastic-windows/boundedwindow/MaximalEndComponentClassifier.h>

namespace sw {
    namespace BoundedWindow {

        template <typename ValueType>
        MaximalEndComponentClassifier<ValueType>::MaximalEndComponentClassifier(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                sw::storage::MaximalEndComponentDecompositionWindowGame<ValueType> const& mecDecompositionGame,
                bool produceScheduler)
                : sw::util::MaximalEndComponentClassifier<ValueType>(mdp, mecDecompositionGame, produceScheduler) {

            if (produceScheduler) {
                this->mecScheduler = std::unique_ptr<storm::storage::Scheduler<ValueType>>
                        (new storm::storage::Scheduler<ValueType>(mdp.getNumberOfStates()));
            }

            for (uint_fast64_t k = 0; k < mecDecompositionGame.size(); ++ k) {
                sw::game::WindowGame<ValueType> const& mecGame = mecDecompositionGame.getGame(k);
                storm::storage::BitVector winningSet = produceScheduler ?
                                                       mecGame.directBoundedProblem(*this->mecScheduler)
                                                       : mecGame.directBoundedProblem();
                // If the winning set in the current MEC game is not empty, then the MEC is classified as good.
                if (not winningSet.empty()) {
                    this->goodMECs.set(k, true);
                    // logical OR
                    this->goodStateSpace |= mecGame.getStateSpace();
                    this->safeStateSpace |= winningSet;
                }
            }
            if (produceScheduler) {
                storm::storage::SparseMatrix<ValueType> backwardTransitions = mdp.getBackwardTransitions();
                for (uint_fast64_t const& k : this->goodMECs) {
                    storm::storage::Scheduler<ValueType> reachSafeStates(mdp.getNumberOfStates());
                    storm::utility::graph::computeSchedulerProb1E(mecDecompositionGame.getGame(k).getStateSpace(), // good states have a probability one of reaching safe states in the same MEC
                                                                  mdp.getTransitionMatrix(), backwardTransitions, // transitions
                                                                  mecDecompositionGame.getGame(k).getStateSpace(), // phi
                                                                  mecDecompositionGame.getGame(k).getStateSpace() & this->safeStateSpace, // psi
                                                                  reachSafeStates); // update scheduler
                    for (uint_fast64_t const& state : mecDecompositionGame.getGame(k).getStateSpace() & ~this->safeStateSpace) {
                        this->mecScheduler->setChoice(reachSafeStates.getChoice(state), state);
                    }
                }
            }
        }

        template class MaximalEndComponentClassifier<double>;
        template class MaximalEndComponentClassifier<storm::RationalNumber>;

    }
}