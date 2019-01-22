//
// Created by Florent Delgrange on 2018-11-28.
//

#include <storm/models/sparse/Mdp.h>
#include <stochastic-windows/fixedwindow/MECsUnfolding.h>
#include <storm/utility/graph.h>
#include <storm/storage/BitVector.h>
#include <storm/storage/Scheduler.h>

#ifndef STORM_FIXEDWINDOW_H
#define STORM_FIXEDWINDOW_H

namespace sw {
    namespace FixedWindow {

        template <typename ValueType>
        class MeanPayoff {

        public:
            MeanPayoff(storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                        std::string const& rewardModelName,
                        uint_fast64_t const& l_max);

        private:
            /*!
             * maximum window size
             */
            uint_fast64_t l_max;
            /*!
             * unfolding of the end components
             */
            MECsUnfoldingMeanPayoff<ValueType> unfoldedECs;
            /*!
             * A good EC is an EC where there exists a strategy almost surely closing windows in at most l_max steps.
             */
            storm::storage::BitVector goodECs;
            /*!
             * safe states are states from which there exists a strategy surely closing windows in at most l_max steps.
             */
            storm::storage::BitVector safeStates;
            /*!
             * good states in unfolded ECs are states in the unfolded end components from which it is always possible to
             * close a window in at most l_max steps.
             */
            std::vector<storm::storage::BitVector> unfoldingWinningSet;

            // storm::storage::Scheduler<ValueType> scheduler;
        };
    }
}


#endif //STORM_FIXEDWINDOW_H
