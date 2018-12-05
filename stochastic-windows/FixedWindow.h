//
// Created by Florent Delgrange on 2018-11-28.
//

#include <storm/models/sparse/Mdp.h>
#include <stochastic-windows/ECsUnfolding.h>
#include <storm/utility/graph.h>
#include <storm/storage/BitVector.h>
#include <storm/storage/Scheduler.h>

#ifndef STORM_FIXEDWINDOW_H
#define STORM_FIXEDWINDOW_H

namespace sw {
    namespace WindowMP {

        template <typename ValueType>
        class FixedWindow {

        public:
            FixedWindow(storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                        std::string const& rewardModelName,
                        uint_fast64_t const& l_max);

        private:
            /*!
             * maximum windows length
             */
            uint_fast64_t l_max;
            /*!
             * unfolding of the end components
             */
            ECsUnfolding<ValueType> unfoldedECs;
            /*!
             * A good EC is an EC where there exists a strategy allowing to always close a window in at most l_max steps.
             */
            storm::storage::BitVector goodECs;
            /*!
             * good states are states from which it is always possible to close a window in at most l_max steps.
             */
            std::vector<storm::storage::BitVector> goodStates;
            /*!
             * good states in unfolded ECs are states in the unfolded end component from which it is always possible to
             * close a window in at most l_max steps.
             */
            std::vector<storm::storage::BitVector> goodStatesInUnfoldedECs;

            // storm::storage::Scheduler<ValueType> scheduler;
        };
    }
}


#endif //STORM_FIXEDWINDOW_H
