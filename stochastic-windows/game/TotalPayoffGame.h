//
// Created by Florent Delgrange on 2019-02-12.
//

#ifndef STORM_TOTALPAYOFFGAME_H
#define STORM_TOTALPAYOFFGAME_H

#include <storm/models/sparse/StandardRewardModel.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/BitVector.h>
#include <stochastic-windows/game/WindowGame.h>
#include <functional>

namespace sw {
    namespace game {

        template <typename ValueType>
        class TotalPayoffGame: public MdpGame<ValueType> {
        public:

            TotalPayoffGame(storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    std::string const &rewardModelName,
                    storm::storage::BitVector const &restrictedStateSpace,
                    storm::storage::BitVector const &enabledActions);

            class iterator {
            public:
                iterator(uint_fast64_t rowBegin, uint_fast64_t rowEnd, storm::storage::BitVector const& enabledActions);
                iterator(typename std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>::const_iterator entriesIteratorBegin,
                         typename std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>::const_iterator entriesIteratorEnd,
                         storm::storage::BitVector const& enabledActions);
                // prefix ++
                iterator& operator++();
                // postfix ++
                iterator operator++(int);
                bool operator!=(iterator const& otherIterator);
                uint_fast64_t operator*();

            private:
                uint_fast64_t currentRow;
                uint_fast64_t rowEnd;
                uint_fast64_t currentColumn;
                typename std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>::const_iterator matrixEntryIterator;
                typename std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>::const_iterator ptr_end;
                bool iterate_on_columns;
                storm::storage::BitVector const& enabledActions;
            };

            class successors {

            public:
                successors(uint_fast64_t state,
                storm::storage::SparseMatrix<ValueType> const& matrix,
                        storm::storage::BitVector const& enabledActions);
                virtual ~successors() = 0;
                virtual iterator begin() = 0;
                virtual iterator end();

            protected:
                uint_fast64_t state;
                storm::storage::SparseMatrix<ValueType> const& matrix;
                storm::storage::BitVector const& enabledActions;
            };

            class successorsP1: public successors {

            public:
                successorsP1(uint_fast64_t state,
                             storm::storage::SparseMatrix<ValueType> const& matrix,
                             storm::storage::BitVector const& enabledActions);
                iterator begin() override;

            };

            class successorsP2: public successors {
            public:
                successorsP2(uint_fast64_t state,
                             storm::storage::SparseMatrix<ValueType> const& matrix,
                             storm::storage::BitVector const& enabledActions);
                iterator begin() override;
            };

            std::vector<ValueType> maxTotalPayoffInf() const;
            std::vector<ValueType> minTotalPayoffSup() const;

        private:
            /**
             * Name of the reward model to consider
             */
            std::string const& rewardModelName;
            /**
             * Reward Model to consider
             */
            storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel;

            std::vector<ValueType> maxTotalPayoffInf(std::function<std::unique_ptr<successors>(uint_fast64_t)> maximizerSuccessors,
                                                     std::function<std::unique_ptr<successors>(uint_fast64_t)> minimizerSuccessors,
                                                     std::function<ValueType(uint_fast64_t, uint_fast64_t)> wMaxToMin,
                                                     std::function<ValueType(uint_fast64_t, uint_fast64_t)> wMinToMax) const;

        };

    }
};



#endif //STORM_TOTALPAYOFFGAME_H
