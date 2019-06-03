//
// Created by Florent Delgrange on 2019-03-21.
//

#ifndef STORM_PARITYRESPONSEGAME_H
#define STORM_PARITYRESPONSEGAME_H

#include <stochastic-windows/game/MdpGame.h>

namespace sw {
    namespace game {

        template <typename ValueType>
        class WeakParityGame: public MdpGame<ValueType> {
        public:

            WeakParityGame(
                    storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    std::string const &rewardModelName,
                    storm::storage::BitVector const &restrictedStateSpace,
                    storm::storage::BitVector const &enabledActions);

            struct WinningRegion {
                GameStates winningSetP1; // winning set for the weak parity objective
                GameStates winningSetP2; // co-winning set for the weak parity objective
            };

            WinningRegion weakParity() const;

            /*!
             * initialize the input BackwardTransitions structure for this MDP game
             * @param backwardTransitions an empty BackwardTransitions structure to initialize
             */
            void initBackwardTransitions(BackwardTransitions &backwardTransitions) const override;

        private:
            /**
             * Reward Model to consider
             */
            std::string const &rewardModelName;
            storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel;

            bool isEven(ValueType priority) const;
        };

    }
}


#endif //STORM_PARITYRESPONSEGAME_H
