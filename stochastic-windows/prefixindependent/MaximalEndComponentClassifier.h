//
// Created by Florent Delgrange on 2019-02-01.
//

#ifndef STORM_MAXIMALENDCOMPONENTCLASSIFIER_H
#define STORM_MAXIMALENDCOMPONENTCLASSIFIER_H

#include <storm/storage/MaximalEndComponent.h>
#include <stochastic-windows/game/WindowGame.h>
#include <stochastic-windows/prefixindependent/MaximalEndComponentDecompositionUnfolding.h>
#include <stochastic-windows/prefixindependent/MaximalEndComponentDecompositionWindowGame.h>
#include <storm/utility/graph.h>

namespace sw {
    namespace util {

    /*!
     * This class allows to classify maximal end components as being good or not.
     */
    template<typename ValueType>
    class MaximalEndComponentClassifier {
    public:

        MaximalEndComponentClassifier(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                storm::storage::MaximalEndComponentDecomposition<ValueType> const& maximalEndComponentDecomposition,
                bool produceScheduler = false);

        virtual ~MaximalEndComponentClassifier() = default;

        /*!
         * Retrieves a vector containing each good maximal end components.
         *
         * @note A good EC (end component) w.r.t. a prefix-independent window objective is an EC for which there exists a
         *       sub-EC where there exists a strategy surely winning this window objective from every of its
         *       state.
         */
        std::vector<std::reference_wrapper<const storm::storage::MaximalEndComponent>> getGoodMaximalEndComponents();
        /*!
         * Retrieves the set of safe states of the input model.
         *
         * @note A state is said to be safe w.r.t. a prefix-independent window objective if and only if there exists a
         *       strategy surely winning this window objective from this state.
         */
        storm::storage::BitVector const& getSafeStateSpace();
        /*!
         * Retrieves the set of good states of the input model (i.e., the union of the state space of each good MEC).
         */
        storm::storage::BitVector const& getGoodStateSpace();
        /*!
         * Retrieves if yes or not a scheduler is initialized to play inside each MEC
         */
        bool hasMaximalEndComponentScheduler() const;
        /*!
         * Retrieves a scheduler allowing to have a probability one to satisfy a window objective inside each MEC
         * if hasMecScheduler is true.
         */
         storm::storage::Scheduler<ValueType> const& getMaximalEndComponentScheduler() const;

        protected:

            storm::storage::MaximalEndComponentDecomposition<ValueType> const& maximalEndComponentDecomposition;
            storm::storage::BitVector safeStateSpace;
            storm::storage::BitVector goodStateSpace;
            storm::storage::BitVector goodMECs;
            std::unique_ptr<storm::storage::Scheduler<ValueType>> mecScheduler;

        };

    }
}


#endif //STORM_MAXIMALENDCOMPONENTCLASSIFIER_H
