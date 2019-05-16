//
// Created by Florent Delgrange on 2019-02-01.
//

#include <stochastic-windows/prefixindependent/MaximalEndComponentClassifier.h>

namespace sw {
    namespace util {

        template<typename ValueType>
        MaximalEndComponentClassifier<ValueType>::MaximalEndComponentClassifier(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                storm::storage::MaximalEndComponentDecomposition<ValueType> const& maximalEndComponentDecomposition,
                bool produceScheduler)
                : maximalEndComponentDecomposition(maximalEndComponentDecomposition),
                  safeStateSpace(mdp.getNumberOfStates(), false),
                  goodStateSpace(mdp.getNumberOfStates(), false),
                  goodMECs(maximalEndComponentDecomposition.size(), false) {}

        template <typename ValueType>
        std::vector<std::reference_wrapper<const storm::storage::MaximalEndComponent>>
        MaximalEndComponentClassifier<ValueType>::getGoodMaximalEndComponents() const {

            std::vector<std::reference_wrapper<const storm::storage::MaximalEndComponent>> goodMaximalEndComponents;
            goodMaximalEndComponents.reserve(this->goodMECs.getNumberOfSetBits());
            for (uint_fast64_t k: this->goodMECs) {
                goodMaximalEndComponents.push_back(std::cref(this->maximalEndComponentDecomposition[k]));
            }
            return goodMaximalEndComponents;
        }

        template <typename ValueType>
        storm::storage::BitVector const& MaximalEndComponentClassifier<ValueType>::getSafeStateSpace() const {
            return this->safeStateSpace;
        }

        template <typename ValueType>
        storm::storage::BitVector const& MaximalEndComponentClassifier<ValueType>::getGoodStateSpace() const {
            return this->goodStateSpace;
        }

        template<typename ValueType>
        const storm::storage::Scheduler<ValueType> &MaximalEndComponentClassifier<ValueType>::getMaximalEndComponentScheduler() const {
            return *this->mecScheduler;
        }

        template<typename ValueType>
        bool MaximalEndComponentClassifier<ValueType>::hasMaximalEndComponentScheduler() const {
            return this->mecScheduler != nullptr;
        }

        template class MaximalEndComponentClassifier<double>;
        template class MaximalEndComponentClassifier<storm::RationalNumber>;

    }
}