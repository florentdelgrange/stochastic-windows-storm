//
// Created by Florent Delgrange on 2019-01-14.
//

#include <storm/storage/BitVector.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/SparseMatrix.h>

#ifndef STORM_UNFOLDING_H
#define STORM_UNFOLDING_H

namespace sw {
    namespace DirectFixedWindow {

        template<typename ValueType>
        struct StateValueWindowSize{
            uint_fast64_t state;
            ValueType currentValue;
            uint_fast64_t currentWindowSize;
        };

        template<typename ValueType>
        class UnfoldingMeanPayoff {
        public:

            UnfoldingMeanPayoff(storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>>& mdp,
                                std::string const& rewardModelName,
                                uint_fast64_t const& l_max,
                                storm::storage::BitVector const& initialStates);

            UnfoldingMeanPayoff(storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>>& mdp,
                    std::string const& rewardModelName,
                    uint_fast64_t const& l_max,
                    storm::storage::BitVector const& initialStates,
                    std::vector<std::vector<uint_fast64_t>> enabledActions);

            storm::storage::SparseMatrix<ValueType>& getMatrix();

            uint_fast64_t getNewIndex(uint_fast64_t state,
                                      ValueType currentSumOfWeights,
                                      uint_fast64_t currentWindowSize);
            /*!
             * Get a vector containing the meaning of each state in the new matrix corresponding to the
             * unfolding of the mdp, expressed as a tuple (s, w, l) where s (state) is the state in the original matrix,
             * w (currentSumOfWeights) is the current sum of weights in the unfolding and l (currentWindowLength) is
             * the current window size in the unfolding.
             */
            std::vector<StateValueWindowSize<ValueType>> getNewStatesMeaning();

        private:

            /*!
             * Maximum window size
             */
            uint_fast64_t l_max;

            storm::storage::SparseMatrix<ValueType> matrix;

            /*!
             * This matrix represents the unfolding of the MDP for the window mean payoff
             * objective.
             */
            storm::storage::SparseMatrix<ValueType>& originalMatrix;

            std::vector<ValueType> stateActionRewardsVector;

            std::vector<std::vector<uint_fast64_t>>& enabledActions;

            /*!
             * This vector contains the index of each original state s in the unfolding regarding to
             * the current window size l and the current sum of weights w in the unfolding of the mdp.
             * usage: the index of the state s in the unfolding of the associated MEC is windowVector[s][l][w]
             */
            std::vector<std::vector<std::unordered_map<ValueType, uint_fast64_t>>> windowVector;

            /*!
             * This vector maps each new state to its pairs of
             * (successor states, probabilities) for each of its enabled actions.
             * Dimensions = (0: state, 1: action, 2: pairs of (s', p) where p is the probability to go to s').
             */
            std::vector<std::vector<std::vector<std::pair<uint_fast64_t, ValueType>>>> newRowGroupEntries;
            /*!
             * Unfold an MDP from a given state. The vectors oldToNewStateMapping and newRowGroupEntries are
             * filled accordingly.
             *
             * @return the index of the input state in the new matrix
             */
            uint_fast64_t unfoldFrom(uint_fast64_t const& state, ValueType const& currentSumOfWeights, uint_fast64_t const& l);

            std::vector<std::vector<uint_fast64_t>> const enableAllActions(storm::storage::SparseMatrix<ValueType> const& originalMatrix);
        };
    }
}


#endif //STORM_UNFOLDING_H
