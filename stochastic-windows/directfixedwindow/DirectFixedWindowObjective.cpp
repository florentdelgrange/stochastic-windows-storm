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
                : mdp(mdp), rewardModelName(rewardModelName), l_max(l_max) {}

        template<typename ValueType>
        DirectFixedWindowObjectiveMeanPayoff<ValueType>::DirectFixedWindowObjectiveMeanPayoff(
                storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const& rewardModelName,
                uint_fast64_t const& l_max)
                : DirectFixedWindowObjective<ValueType>(mdp, rewardModelName, l_max) {}

        template<typename ValueType>
        DirectFixedWindowObjectiveParity<ValueType>::DirectFixedWindowObjectiveParity(
                storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const& rewardModelName,
                uint_fast64_t const& l_max)
                : DirectFixedWindowObjective<ValueType>(mdp, rewardModelName, l_max) {}

        template<typename ValueType>
        WindowUnfolding<ValueType> DirectFixedWindowObjectiveMeanPayoff<ValueType>::performUnfolding(
                storm::storage::BitVector const &initialStates) const {
            return WindowUnfoldingMeanPayoff<ValueType>(this->mdp, this->rewardModelName, this->l_max, initialStates);
        }

        template<typename ValueType>
        WindowUnfolding<ValueType> DirectFixedWindowObjectiveParity<ValueType>::performUnfolding(
                storm::storage::BitVector const &initialStates) const {
            return WindowUnfoldingParity<ValueType>(this->mdp, this->rewardModelName, this->l_max, initialStates);
        }

        template<typename ValueType>
        const storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &
        DirectFixedWindowObjective<ValueType>::getMdp() const {
            return mdp;
        }

        template<typename ValueType>
        const std::string &DirectFixedWindowObjective<ValueType>::getRewardModelName() const {
            return rewardModelName;
        }

        template<typename ValueType>
        uint_fast64_t DirectFixedWindowObjective<ValueType>::getMaximumWindowSize() {
            return l_max;
        }

        template<typename ValueType>
        std::vector<ValueType> performMaxProb(storm::storage::BitVector const& phiStates,
                DirectFixedWindowObjective<ValueType> const& dfwObjective,
                bool useMecBasedTechnique) {
            std::vector<ValueType> result(dfwObjective.getMdp().getNumberOfStates(), 0);
            WindowUnfolding<ValueType> unfolding = dfwObjective.performUnfolding(phiStates);
            storm::storage::BitVector psiStates(unfolding.getMatrix().getRowGroupCount(), true);
            psiStates.set(0, false); // 0 is the index of ⊥ in the unfolding, the state we want to avoid
            std::vector<ValueType> resultInUnfolding =
                    storm::modelchecker::helper::SparseMdpPrctlHelper<ValueType>().computeGloballyProbabilities(
                            storm::Environment(),
                            storm::solver::SolveGoal<ValueType>(false), // we want to maximize the probability of staying in ¬⊥
                            unfolding.getMatrix(),
                            unfolding.getMatrix().transpose(true),
                            psiStates,
                            useMecBasedTechnique);
            std::vector<StateValueWindowSize<ValueType>> meanings = unfolding.getNewStatesMeaning();
            for (uint_fast64_t state: phiStates) {
                result[state] = resultInUnfolding[unfolding.getInitialState(state)];
                std::cout << "initial state for " << state << " = ("
                    << meanings[unfolding.getInitialState(state)].state << ", "
                    << meanings[unfolding.getInitialState(state)].currentValue << ", "
                    << meanings[unfolding.getInitialState(state)].currentWindowSize << ")" << std::endl;
            }
            return result;
        }

        template<typename ValueType>
        ValueType performMaxProb(uint_fast64_t state,
                DirectFixedWindowObjective<ValueType> const& dfwObjective,
                bool useMecBasedTechnique) {
            storm::storage::BitVector phiStates(dfwObjective.getMdp().getNumberOfStates(), false);
            phiStates.set(state, true);
            return performMaxProb<ValueType>(phiStates, dfwObjective, useMecBasedTechnique)[state];
        }

        template class DirectFixedWindowObjective<double>;
        template class DirectFixedWindowObjective<storm::RationalNumber>;
        template class DirectFixedWindowObjectiveMeanPayoff<double>;
        template class DirectFixedWindowObjectiveMeanPayoff<storm::RationalNumber>;
        template class DirectFixedWindowObjectiveParity<double>;

        template std::vector<double> performMaxProb<double>(storm::storage::BitVector const& phiStates, DirectFixedWindowObjective<double> const& dfwObjective, bool useMecBasedTechnique);
        template std::vector<storm::RationalNumber> performMaxProb<storm::RationalNumber>(storm::storage::BitVector const& phiStates, DirectFixedWindowObjective<storm::RationalNumber> const& dfwObjective, bool useMecBasedTechnique);
        template double performMaxProb<double>(uint_fast64_t state, DirectFixedWindowObjective<double> const& dfwObjective, bool useMecBasedTechnique);
        template storm::RationalNumber performMaxProb<storm::RationalNumber>(uint_fast64_t state, DirectFixedWindowObjective<storm::RationalNumber> const& dfwObjective, bool useMecBasedTechnique);

    }
}