//
// Created by Florent Delgrange on 2019-01-25.
//

#ifndef STORM_WINDOWGAME_H
#define STORM_WINDOWGAME_H

namespace sw {
    namespace Game {

        template<typename ValueType>
        class WindowGame {
        public:

            /**
             * Consider the MDP as a game to synthesize strategies for window objectives.
             *
             * @param mdp the model to consider as a game
             * @param rewardModelName the name of the reward model to consider
             * @param l_max the maximum window size
             * @param enabledActions the set of actions to consider in the MDP-game
             */
            WindowGame(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const& rewardModelName,
                    uint_fast64_t const& l_max,
                    storm::storage::BitVector const& enabledActions);

            /**
             * Compute the winning set of states from which there exists a strategy allowing to surely close a window in
             * l_max steps or less
             *
             * @return the winning set for the GoodWindow objective
             */
            virtual storm::storage::BitVector goodWin() = 0;

        protected:
            /**
             * MDP to consider as a game
             */
            storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
            /**
             * Reward Model of the MDP
             */
            storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel;
            /**
             * maximum window size
             */
            uint_fast64_t const& l_max,
            /**
             * set of actions to consider in the MDP-game
             */
            storm::storage::BitVector const& enabledActions;

        };

        template<typename ValueType>
        class WindowGameMeanPayoff: public WindowGame {
        public:

            /**
             * Consider the MDP as a game to synthesize strategies for window mean-payoff objectives.
             *
             * @param mdp the model to consider as a game
             * @param rewardModelName the name of the reward model to consider
             * @param l_max the maximum window size
             * @param enabledActions the set of actions to consider in the MDP-game
             */
            WindowGameMeanPayoff(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const& rewardModelName,
                    uint_fast64_t const& l_max,
                    storm::storage::BitVector const& enabledActions);

            storm::storage::BitVector goodWin() override;

        };

        template<typename ValueType>
        class WindowGameParity: public WindowGame {
        public:

            /**
             * Consider the MDP as a game to synthesize strategies for window parity objectives.
             *
             * @param mdp the model to consider as a game
             * @param rewardModelName the name of the reward model to consider
             * @param l_max the maximum window size
             * @param enabledActions the set of actions to consider in the MDP-game
             */
            WindowGameMeanParity(
                    storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                    std::string const& rewardModelName,
                    uint_fast64_t const& l_max,
                    storm::storage::BitVector const& enabledActions);

            storm::storage::BitVector goodWin() override;

        };
}


#endif //STORM_WINDOWGAME_H
