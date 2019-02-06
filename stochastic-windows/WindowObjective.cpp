//
// Created by Florent Delgrange on 2019-02-05.
//

#include "WindowObjective.h"

namespace sw {

    template<typename ValueType>
    WindowObjective<ValueType>::~WindowObjective() = default;

    template<typename ValueType>
    WindowObjective<ValueType>::WindowObjective(
            storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
            std::string const &rewardModelName)
            : mdp(mdp), rewardModelName(rewardModelName) {}

    template<typename ValueType>
    storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& WindowObjective<ValueType>::getMdp() const {
        return this->mdp;
    }

    template<typename ValueType>
    std::string const& WindowObjective<ValueType>::getRewardModelName() const {
        return this->rewardModelName;
    }

    template class WindowObjective<double>;
    template class WindowObjective<storm::RationalNumber>;

}