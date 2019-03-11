//
// Created by Florent Delgrange on 2019-02-01.
//

#include <stochastic-windows/prefixindependent/MaximalEndComponentClassifier.h>

namespace sw {
    namespace util {

        template<typename ValueType>
        MaximalEndComponentClassifier<ValueType>::MaximalEndComponentClassifier(
                storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                storm::storage::MaximalEndComponentDecomposition<ValueType> const& maximalEndComponentDecomposition)
                : maximalEndComponentDecomposition(maximalEndComponentDecomposition),
                  safeStateSpace(mdp.getNumberOfStates(), false),
                  goodStateSpace(mdp.getNumberOfStates(), false),
                  goodMECs(maximalEndComponentDecomposition.size(), false) {}

        template <typename ValueType>
        std::vector<std::reference_wrapper<const storm::storage::MaximalEndComponent>>
        MaximalEndComponentClassifier<ValueType>::getGoodMaximalEndComponents() {

            std::vector<std::reference_wrapper<const storm::storage::MaximalEndComponent>> goodMaximalEndComponents;
            goodMaximalEndComponents.reserve(this->goodMECs.getNumberOfSetBits());
            for (uint_fast64_t k: this->goodMECs) {
                goodMaximalEndComponents.push_back(std::cref(this->maximalEndComponentDecomposition[k]));
            }
            return goodMaximalEndComponents;
        }

        template <typename ValueType>
        storm::storage::BitVector const& MaximalEndComponentClassifier<ValueType>::getSafeStateSpace() {
            return this->safeStateSpace;
        }

        template <typename ValueType>
        storm::storage::BitVector const& MaximalEndComponentClassifier<ValueType>::getGoodStateSpace() {
            return this->goodStateSpace;
        }

        template class MaximalEndComponentClassifier<double>;
        template class MaximalEndComponentClassifier<storm::RationalNumber>;

    }
}