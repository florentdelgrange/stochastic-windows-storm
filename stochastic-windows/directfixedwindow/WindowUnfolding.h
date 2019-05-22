//
// Created by Florent Delgrange on 2019-01-14.
//

#include <storm/storage/BitVector.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/utility/constants.h>
#include <storm/storage/memorystructure/MemoryStructure.h>
#include <storm/storage/memorystructure/MemoryStructureBuilder.h>

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

        template <typename ValueType>
        struct WindowMemory {
            /*!
             * Memory structure recording information refining the original MDP into the unfolding of this MDP
             */
            std::unique_ptr<storm::storage::MemoryStructure> memoryStructure;
            /*!
             * Mapping that links each state of the unfolding to a memory state of the memory structure above.
             */
            std::vector<uint_fast64_t> unfoldingToMemoryStatesMapping;
        };

        template<typename ValueType>
        class WindowUnfolding {
        public:

            WindowUnfolding(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const& rewardModelName,
                    uint_fast64_t const& l_max,
                    storm::storage::BitVector const& enabledActions);

            WindowUnfolding(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const& rewardModelName,
                    uint_fast64_t const& l_max);
            /*
             * Make destructor virtual to allow deleting objects through pointer to base class(es).
             */
            virtual ~WindowUnfolding() = default;

            storm::storage::SparseMatrix<ValueType> const& getMatrix() const;

            uint_fast64_t getNewIndex(uint_fast64_t state, ValueType currentSumOfWeights, uint_fast64_t currentWindowSize) const;
            /*!
             * Gets a vector containing the meaning of each state in the new matrix corresponding to the
             * unfolding of the mdp, expressed as a tuple (s, v, l) where s (state) is the state in the original matrix,
             * v (value) is the current value (e.g., current sum of weights or current minimal priority seen) in the
             * unfolding and l (currentWindowSize) is the current window size in the unfolding.
             */
            std::vector<StateValueWindowSize<ValueType>> getNewStatesMeaning() const;

            /*!
             * Retrieves a mapping between actions of the unfolding and the original MDP
             */
            std::vector<uint_fast64_t> newToOldActionsMapping(std::vector<StateValueWindowSize<ValueType>> const& newStatesMeaning) const;

            /*!
             * Gets the index in the unfolding of the input initial state.
             *
             * @param originalInitialState index of an initial state in the original matrix
             * @return the index of this initial state in the unfolding if it exists, 0 otherwise
             */
            virtual uint_fast64_t getInitialState(uint_fast64_t originalInitialState) const = 0;

            /**
             * Generates (i) a memory structure for the original MDP representing this unfolding, i.e.,
             * the product of the original MDP and the memory structure generated retrieves this unfolding;
             * and (ii) the mapping of each state of this unfolding to a memory state of the memory structure.
             */
            WindowMemory<ValueType> generateMemory(bool setLabels=true) const;

            /*!
             * Retrieves the set of actions in the original MDP considered for this unfolding
             */
            storm::storage::BitVector const& getOriginalEnabledActions() const;

        protected:

            /*!
             * Maximum window size
             */
            uint_fast64_t l_max;

            /*!
             * Original MDP
             */
            storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &originalModel;

            /*!
             * Matrix of the unfolding of the original MDP
             */
            storm::storage::SparseMatrix<ValueType> matrix;

            /*!
             * Matrix of the MDP to unfold
             */
            storm::storage::SparseMatrix<ValueType> const& originalMatrix;

            storm::storage::BitVector const enabledActions;

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
            virtual uint_fast64_t unfoldFrom(uint_fast64_t const& state, ValueType const& value, uint_fast64_t const& l) = 0;

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
            virtual ValueType initialStateValue(uint_fast64_t initialState) const = 0;
        };

        template<typename ValueType>
        class WindowUnfoldingMeanPayoff : public WindowUnfolding<ValueType> {
        public:

            WindowUnfoldingMeanPayoff(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &initialStates,
                    storm::storage::BitVector const& enabledActions);

            WindowUnfoldingMeanPayoff(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &initialStates);

            uint_fast64_t getInitialState(uint_fast64_t originalInitialState) const override;

        protected:
            uint_fast64_t unfoldFrom(uint_fast64_t const &state, ValueType const &value, uint_fast64_t const &l) override;
            ValueType initialStateValue(uint_fast64_t state) const override;
        };

        template<typename ValueType>
        class WindowUnfoldingParity : public WindowUnfolding<ValueType> {
        public:

            WindowUnfoldingParity(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &initialStates,
                    storm::storage::BitVector const& enabledActions);

            WindowUnfoldingParity(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &initialStates);

            uint_fast64_t getInitialState(uint_fast64_t originalInitialState) const override;

        protected:
            uint_fast64_t unfoldFrom(uint_fast64_t const &state, ValueType const &value, uint_fast64_t const &l) override;
            ValueType initialStateValue(uint_fast64_t state) const override;
        private:
            bool isEven(ValueType const& priority);
        };
    }
}


#endif //STORM_UNFOLDING_H
