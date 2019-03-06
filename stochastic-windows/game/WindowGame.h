//
// Created by Florent Delgrange on 2019-01-25.
//

#include <storm/storage/BitVector.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/SparseMatrix.h>
#include <stochastic-windows/game/MdpGame.h>

#ifndef STORM_WINDOWGAME_H
#define STORM_WINDOWGAME_H

namespace sw {
    namespace game {

        template<typename ValueType>
        class WindowGame: public MdpGame<ValueType> {
        public:

            /**
             * Consider the MDP as a game for window objectives.
             *
             * @param mdp the model to consider as a game
             * @param rewardModelName the name of the reward model to consider
             * @param l_max the maximum window size
             * @param restrictedStateSpace the set of states to consider in the MDP-game forming the state space of the
             *        Player 1 in the MDP-game.
             * @param enabledActions the set of actions to consider in the MDP-game forming the state space of the
             *        Player 2 in the MDP-game.
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
            virtual storm::storage::BitVector goodWin() const = 0;

            /**
             * Compute the winning set of states from which there exists a strategy allowing to continually surely close
             * all windows in l_max steps or less.
             *
             * @return the winning set for the DirectFixedWindow objective
             */
            storm::storage::BitVector directFW() const;

            /*!
             * Retrieves the considered Player 1 state space of this window game.
             */
            storm::storage::BitVector const& getStateSpace() const;

            /*!
             * Restrict this WindowGame to the safe part of the input set of states.
             * In the resulting sub-MDP-game, all choices ensure to always visit input set of states.
             * @param safeStates set of states in which the new restricted WindowGame will be ensured to stay in it.
             * @return a pointer to a new WindowGame representing the safe part of this WindowGame.
             */
            std::unique_ptr<WindowGame<ValueType>> restrictToSafePart(storm::storage::BitVector const& safeStates) const;

            storm::storage::BitVector boundedProblem() const;
            virtual GameStates unbOpenWindow() const;

        protected:

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

            virtual std::unique_ptr<WindowGame<ValueType>> restrictToSafePart(storm::storage::BitVector const& safeStates,
                    BackwardTransitions& backwardTransitions) const = 0;

            storm::storage::BitVector directFW(BackwardTransitions &backwardTransitions) const;

            virtual std::unique_ptr<WindowGame<ValueType>> unsafeRestrict(
                    storm::storage::BitVector const &restrictedStateSpace) const;


        };

        template<typename ValueType>
        class WindowMeanPayoffGame : public WindowGame<ValueType> {
        public:

            /**
             * Consider the MDP as a game for window mean-payoff objectives.
             * Linked window algorithms' implementations for the particular case of MDPs as games of the ones from
             * Chatterjee K., Doyen L., Randour M., Raskin JF. (2013) Looking at Mean-Payoff and Total-Payoff through Windows.
             * In: Van Hung D., Ogawa M. (eds) Automated Technology for Verification and Analysis. Lecture Notes in Computer Science, vol 8172. Springer, Cham
             * arXiv:1302.4248v3
             * @param mdp the model to consider as a game
             * @param rewardModelName the name of the reward model to consider
             * @param l_max the maximum window size
             * @param restrictedStateSpace the set of states to consider in the MDP-game forming the state space of the
             *        Player 1 in the MDP-game.
             * @param enabledActions the set of actions to consider in the MDP-game forming the state space of the
             *        Player 2 in the MDP-game.
             */
            WindowMeanPayoffGame(
                    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    std::string const &rewardModelName,
                    uint_fast64_t const &l_max,
                    storm::storage::BitVector const &restrictedStateSpace,
                    storm::storage::BitVector const &enabledActions);


            storm::storage::BitVector goodWin() const override;
            GameStates unbOpenWindow() const override;

        protected:

            std::unique_ptr<WindowGame<ValueType>> restrictToSafePart(storm::storage::BitVector const &safeStates,
                    BackwardTransitions& backwardTransitions) const override;

            std::unique_ptr<WindowGame<ValueType>> unsafeRestrict(
                    storm::storage::BitVector const &restrictedStateSpace) const override ;
        };

    }
}


#endif //STORM_WINDOWGAME_H
