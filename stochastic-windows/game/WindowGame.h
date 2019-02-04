//
// Created by Florent Delgrange on 2019-01-25.
//

#include <storm/storage/BitVector.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/SparseMatrix.h>

#ifndef STORM_WINDOWGAME_H
#define STORM_WINDOWGAME_H

namespace sw {
    namespace Game {

        struct BackwardTransitions {
            std::vector<std::forward_list<uint_fast64_t>> statesPredecessors;
            std::vector<uint_fast64_t> actionsPredecessor;
            std::vector<uint_fast64_t> numberOfEnabledActions;
        };

        template<typename ValueType>
        class WindowGame {
        public:

            /**
             * Consider the MDP as a game for window objectives.
             *
             * @param mdp the model to consider as a game
             * @param rewardModelName the name of the reward model to consider
             * @param l_max the maximum window size
             * @param restrictedStateSpace the set of states to consider in the MDP-game
             * @param enabledActions the set of actions to consider in the MDP-game
             * @note strong assumption: choosing an enabled action must always lead to a state of the restricted state space.
             */
            WindowGame(
                    storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &restrictedStateSpace,
                    storm::storage::BitVector const &enabledActions);

            /*
             * Make destructor virtual to allow deleting objects through pointer to base class(es).
             */
            virtual ~WindowGame() = default;

            /**
             * Compute the winning set of states from which there exists a strategy allowing to surely close a window in
             * l_max steps or less.
             *
             * @return the winning set for the GoodWindow objective
             */
            virtual storm::storage::BitVector goodWin() = 0;

            /**
             * Compute the winning set of states from which there exists a strategy allowing to continually surely close
             * all windows in l_max steps or less.
             *
             * @return the winning set for the DirectFixedWindow objective
             */
            storm::storage::BitVector directFWMP();

            /*!
             * Restrict this WindowGame to the safe part of the input state space.
             * In the resulting sub-MDP-game, all choices ensure to always visit input safe states.
             * @param safeStates set of states in which the new restricted WindowGame will be ensured to stay in it.
             * @return a pointer to a new WindowGame representing the safe part of this WindowGame.
             */
            std::unique_ptr<WindowGame<ValueType>> restrictToSafePart(storm::storage::BitVector const& safeStates);

        protected:

            /**
             * MDP to consider as a game
             */
            storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp;
            /**
             * matrix of the MDP to consider as a game
             */
            storm::storage::SparseMatrix<ValueType> const &matrix;
            /**
             * Name of the reward model to consider
             */
            std::string const& rewardModelName;
            /**
             * Reward Model to consider
             */
            storm::models::sparse::StandardRewardModel<ValueType> const &rewardModel;
            /**
             * maximum window size
             */
            uint_fast64_t const &l_max;
            /**
             * set of states to consider in the MDP-game
             */
            storm::storage::BitVector const restrictedStateSpace;
            /**
             * set of actions to consider in the MDP-game
             */
            storm::storage::BitVector const enabledActions;

            /**
             * Compute the set of successor states of each (state, action) pair w.r.t. the restricted state space and
             * enabled actions considered.
             *
             * @note deprecated since we consider that all choice from the enabled actions leads to a state of the
             * restricted state space.
             */
            std::vector<storm::storage::BitVector> getSuccessorStates();

            virtual std::unique_ptr<WindowGame<ValueType>> restrictToSafePart(storm::storage::BitVector const& safeStates,
                    BackwardTransitions& backwardTransitions) = 0;

            /*!
             * initialize the input BackwardTransitions structure for this window game
             * @param backwardTransitions an empty BackwardTransitions structure to initialize
             */
            void initBackwardTransitions(BackwardTransitions& backwardTransitions);

            storm::storage::BitVector directFWMP(BackwardTransitions& backwardTransitions);

        };

        template<typename ValueType>
        class WindowMeanPayoffGame : public WindowGame<ValueType> {
        public:

            /**
             * Consider the MDP as a game for window mean-payoff objectives.
             *
             * @param mdp the model to consider as a game
             * @param rewardModelName the name of the reward model to consider
             * @param l_max the maximum window size
             * @param restrictedStateSpace the set of states to consider in the MDP-game
             * @param enabledActions the set of actions to consider in the MDP-game
             */
            WindowMeanPayoffGame(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &restrictedStateSpace,
                    storm::storage::BitVector const &enabledActions);


            storm::storage::BitVector goodWin() override;

        protected:

            std::unique_ptr<WindowGame<ValueType>> restrictToSafePart(storm::storage::BitVector const &safeStates,
                    BackwardTransitions& backwardTransitions) override;

        };

    }
}


#endif //STORM_WINDOWGAME_H
