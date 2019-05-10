//
// Created by Florent Delgrange on 2019-01-22.
//

#include "DirectFixedWindowObjective.h"

namespace sw {
    namespace DirectFixedWindow {

        template<typename ValueType>
        DirectFixedWindowObjective<ValueType>::DirectFixedWindowObjective(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const &rewardModelName,
                uint_fast64_t const &l_max)
                : WindowObjective<ValueType>(mdp, rewardModelName), l_max(l_max) {}

        template<typename ValueType>
        DirectFixedWindowMeanPayoffObjective<ValueType>::DirectFixedWindowMeanPayoffObjective(
                storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const& rewardModelName,
                uint_fast64_t const& l_max)
                : DirectFixedWindowObjective<ValueType>(mdp, rewardModelName, l_max) {}

        template<typename ValueType>
        DirectFixedWindowParityObjective<ValueType>::DirectFixedWindowParityObjective(
                storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const& rewardModelName,
                uint_fast64_t const& l_max)
                : DirectFixedWindowObjective<ValueType>(mdp, rewardModelName, l_max) {}

        template<typename ValueType>
        std::unique_ptr<WindowUnfolding<ValueType>> DirectFixedWindowMeanPayoffObjective<ValueType>::performUnfolding(
                storm::storage::BitVector const &initialStates) const {
            return std::unique_ptr<WindowUnfolding<ValueType>>(new WindowUnfoldingMeanPayoff<ValueType>(this->mdp, this->rewardModelName, this->l_max, initialStates));
        }

        template<typename ValueType>
        std::unique_ptr<WindowUnfolding<ValueType>> DirectFixedWindowParityObjective<ValueType>::performUnfolding(
                storm::storage::BitVector const &initialStates) const {
            return std::unique_ptr<WindowUnfolding<ValueType>>(new WindowUnfoldingParity<ValueType>(this->mdp, this->rewardModelName, this->l_max, initialStates));
        }

        template<typename ValueType>
        uint_fast64_t DirectFixedWindowObjective<ValueType>::getMaximumWindowSize() const {
            return l_max;
        }

        template<typename ValueType>
        sw::storage::ValuesAndScheduler<ValueType> performMaxProb(storm::storage::BitVector const& phiStates,
                DirectFixedWindowObjective<ValueType> const& dfwObjective,
                bool produceScheduler) {
            std::vector<ValueType> result(dfwObjective.getMdp().getNumberOfStates(), 0);
            std::unique_ptr<WindowUnfolding<ValueType>> unfolding = dfwObjective.performUnfolding(phiStates);
            storm::storage::BitVector psiStates(unfolding->getMatrix().getRowGroupCount(), true);
            psiStates.set(0, false); // 0 is the index of the sink state ⊥ that we want to avoid in the unfolding
            if (produceScheduler) {
                storm::storage::SparseMatrix<ValueType> transposedMatrix = unfolding->getMatrix().transpose(true);
                storm::storage::MaximalEndComponentDecomposition<ValueType> mecDecomposition(unfolding->getMatrix(), transposedMatrix, psiStates);
                storm::storage::BitVector statesInPsiMecs(unfolding->getMatrix().getRowGroupCount());
                for (auto const& mec : mecDecomposition) {
                    for (auto const& stateActionsPair : mec) {
                        statesInPsiMecs.set(stateActionsPair.first, true);
                    }
                }
                storm::modelchecker::helper::MDPSparseModelCheckingHelperReturnType<ValueType>
                resultInUnfolding = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>().computeUntilProbabilities(
                        storm::Environment(),
                        storm::solver::OptimizationDirection::Maximize, // maximize the probability
                        unfolding->getMatrix(),
                        transposedMatrix,
                        psiStates,
                        statesInPsiMecs,
                        false, // quantitative
                        true); // produce scheduler
                // result in the unfolding to result in the original MDP
                for (uint_fast64_t state: phiStates) {
                    result[state] = resultInUnfolding.values[unfolding->getInitialState(state)];
                }
                WindowMemory<ValueType> windowMemory = unfolding->generateMemory();
                std::cout << "MEMORY" << windowMemory.memoryStructure->toString() << std::endl;
                std::unique_ptr<storm::storage::Scheduler<ValueType>>
                scheduler = std::unique_ptr<storm::storage::Scheduler<ValueType>>(
                        new storm::storage::Scheduler<ValueType>(dfwObjective.getMdp().getNumberOfStates(), *windowMemory.memoryStructure)
                );
                std::vector<StateValueWindowSize<ValueType>> newStatesMeaning = unfolding->getNewStatesMeaning();
                std::vector<uint_fast64_t> actionsMapping = unfolding->newToOldActionsMapping(newStatesMeaning);
                for (uint_fast64_t unfoldingState = 1; unfoldingState < unfolding->getMatrix().getRowGroupCount(); ++ unfoldingState) {
                    uint_fast64_t state = newStatesMeaning[unfoldingState].state;
                    uint_fast64_t memoryState = windowMemory.unfoldingToMemoryStatesMapping[unfoldingState];
                    scheduler->setChoice(actionsMapping[resultInUnfolding.scheduler->getChoice(unfoldingState).getDeterministicChoice()], state, memoryState);
                }
                for (uint_fast64_t state = 0; state < dfwObjective.getMdp().getNumberOfStates(); ++ state) {
                    for (uint_fast64_t memoryState = 0; memoryState < scheduler->getNumberOfMemoryStates(); ++ memoryState) {
                        if (not scheduler->getChoice(state, memoryState).isDefined()) {
                            scheduler->setChoice(dfwObjective.getMdp().getTransitionMatrix().getRowGroupIndices()[state], state, memoryState);
                        }
                    }
                }
                return sw::storage::ValuesAndScheduler<ValueType>(std::move(result), std::move(scheduler));
            }
            else {
                std::vector<ValueType>
                resultInUnfolding = storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>().computeGloballyProbabilities(
                        storm::Environment(),
                        storm::solver::SolveGoal<ValueType>(false), // we want to maximize the probability of staying in ¬⊥
                        unfolding->getMatrix(),
                        unfolding->getMatrix().transpose(true),
                        psiStates,
                        false, // quantitative
                        true); // use MEC based technique
                for (uint_fast64_t state: phiStates) {
                    result[state] = resultInUnfolding[unfolding->getInitialState(state)];
                }
                return sw::storage::ValuesAndScheduler<ValueType>(std::move(result));
            }
        }

        template class DirectFixedWindowObjective<double>;
        template class DirectFixedWindowObjective<storm::RationalNumber>;
        template class DirectFixedWindowMeanPayoffObjective<double>;
        template class DirectFixedWindowMeanPayoffObjective<storm::RationalNumber>;
        template class DirectFixedWindowParityObjective<double>;

        template sw::storage::ValuesAndScheduler<double> performMaxProb<double>(storm::storage::BitVector const& phiStates, DirectFixedWindowObjective<double> const& dfwObjective, bool produceScheduler);
        template sw::storage::ValuesAndScheduler<storm::RationalNumber> performMaxProb<storm::RationalNumber>(storm::storage::BitVector const& phiStates, DirectFixedWindowObjective<storm::RationalNumber> const& dfwObjective, bool produceScheduler);

    }
}