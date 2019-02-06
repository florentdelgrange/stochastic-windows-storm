//
// Created by Florent Delgrange on 2019-02-05.
//

#ifndef STORM_WINDOWOBJECTIVE_H
#define STORM_WINDOWOBJECTIVE_H

#include <storm/models/sparse/Mdp.h>
#include <storm/models/sparse/StandardRewardModel.h>

namespace sw {

    template<typename ValueType>
    class WindowObjective {
    public:

        WindowObjective(
                storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp,
                std::string const& rewardModelName);

        virtual ~WindowObjective() = 0;

        storm::models::sparse::Mdp<ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const& getMdp() const;
        std::string const& getRewardModelName() const;

    protected:

        storm::models::sparse::Mdp<ValueType,storm::models::sparse::StandardRewardModel<ValueType>> const& mdp;
        std::string const& rewardModelName;

    };
}


#endif //STORM_WINDOWOBJECTIVE_H
