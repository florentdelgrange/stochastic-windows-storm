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
        class WindowUnfolding {
        public:

            WindowUnfolding(storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>>& mdp,
                            std::string const& rewardModelName,
                            uint_fast64_t const& l_max,
                            storm::storage::BitVector enabledActions);

            WindowUnfolding(storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>>& mdp,
                            std::string const& rewardModelName,
                            uint_fast64_t const& l_max);
            /*
             * Make destructor virtual to allow deleting objects through pointer to base classe(s).
             */
            virtual ~WindowUnfolding() {}

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

        protected:

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

            storm::storage::BitVector enabledActions;

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

            storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel;

            /*!
             * Unfold an MDP from a given state. The vectors windowVector and newRowGroupEntries are
             * filled accordingly.
             *
             * @return the index of the input state in the new matrix
             */
            virtual uint_fast64_t unfoldFrom(uint_fast64_t const& state, ValueType const& value, uint_fast64_t const& l);

            /*!
             * Constructs the matrix representing the unfolding of the original matrix for the window objective from the
             * initial states given as parameter
             *
             * @param initialStates states of the original MDP from which it will be unfolded
             */
            void generateMatrix(storm::storage::BitVector const &initialStates);

            /*!
             * Gives the initial value of the input initial state in the unfolding
             */
            virtual ValueType initialStateValue(uint_fast64_t initialState);
        };

        template<typename ValueType>
        class WindowUnfoldingMeanPayoff : public WindowUnfolding<ValueType> {
        public:

            WindowUnfoldingMeanPayoff(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &initialStates,
                    storm::storage::BitVector enabledActions);

            WindowUnfoldingMeanPayoff(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &initialStates);

        protected:
            uint_fast64_t unfoldFrom(uint_fast64_t const &state, ValueType const &value, uint_fast64_t const &l);
            ValueType initialStateValue(uint_fast64_t state);
        };
    }
}


#endif //STORM_UNFOLDING_H
