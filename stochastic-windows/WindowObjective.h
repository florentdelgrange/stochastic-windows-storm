//
// Created by Florent Delgrange on 2019-02-05.
//

#ifndef STORM_WINDOWOBJECTIVE_H
#define STORM_WINDOWOBJECTIVE_H

#include <storm/models/sparse/Mdp.h>
#include <storm/models/sparse/StandardRewardModel.h>

namespace sw {

    namespace storage {

        template <typename ValueType>
        struct ValuesAndScheduler {
            explicit ValuesAndScheduler(std::vector<ValueType> &&values, std::unique_ptr<storm::storage::Scheduler<ValueType>>&& scheduler = nullptr)
            : values(std::move(values)), scheduler(std::move(scheduler)) {}
            ValuesAndScheduler(ValuesAndScheduler<ValueType> &&other) noexcept : values(std::move(other.values)), scheduler(std::move(other.scheduler)) {}
            ValuesAndScheduler<ValueType>& operator=(const sw::storage::ValuesAndScheduler<double>& other) {
                this->values = other.values;
                this->scheduler = std::unique_ptr<storm::storage::Scheduler<ValueType>>(new storm::storage::Scheduler<ValueType>(*this->scheduler));
                return *this;
            }
            // The values computed for input states.
            std::vector<ValueType> values;
            // A scheduler, if it was computed.
            std::unique_ptr<storm::storage::Scheduler<ValueType>> scheduler;
        };

        template <typename ValueType>
        struct GoodStateSpaceAndScheduler {
            GoodStateSpaceAndScheduler(storm::storage::BitVector &&goodStateSpace, storm::storage::Scheduler<ValueType>&& scheduler)
            : goodStateSpace(std::move(goodStateSpace)), scheduler(std::move(scheduler)) {}
            storm::storage::BitVector goodStateSpace;
            storm::storage::Scheduler<ValueType> scheduler;
        };

    }

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
