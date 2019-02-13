//
// Created by Florent Delgrange on 2019-02-12.
//

#include "TotalPayoffGame.h"

namespace sw {
    namespace game {

        template<typename ValueType>
        TotalPayoffGame<ValueType>::TotalPayoffGame(
                const storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> &mdp,
                const std::string &rewardModelName,
                const storm::storage::BitVector &restrictedStateSpace,
                const storm::storage::BitVector &enabledActions)
                : MdpGame<ValueType>(mdp, restrictedStateSpace, enabledActions),
                  rewardModelName(rewardModelName),
                  rewardModel(mdp.getRewardModel(rewardModelName)) {}

        template<typename ValueType>
        std::vector<ValueType> TotalPayoffGame<ValueType>::maxTotalPayoffInf() const {
            std::vector<ValueType> const& weights = rewardModel.getStateActionRewardVector();
            return maxTotalPayoffInf(
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new successorsP1(state, this->matrix, this->enabledActions) ); },
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new successorsP2(state, this->matrix, this->enabledActions) ); },
                    [&](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return weights[s_prime]; },
                    [](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return storm::utility::zero<ValueType>(); });
        }

        template<typename ValueType>
        std::vector<ValueType> TotalPayoffGame<ValueType>::minTotalPayoffSup() const {
            std::vector<ValueType> weights = rewardModel.getStateActionRewardVector();
            std::for_each(weights.begin(), weights.end(), [](ValueType w) -> ValueType { return w * -1; });
            return maxTotalPayoffInf(
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new successorsP2(state, this->matrix, this->enabledActions) ); },
                    [&](uint_fast64_t state) -> std::unique_ptr<successors> {
                        return std::unique_ptr<successors>( new successorsP1(state, this->matrix, this->enabledActions) ); },
                    [](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return storm::utility::zero<ValueType>(); },
                    [&](uint_fast64_t s, uint_fast64_t s_prime) -> ValueType { return weights[s_prime]; });
        }

        template<typename ValueType>
        std::vector<ValueType> TotalPayoffGame<ValueType>::maxTotalPayoffInf(
                std::function<std::unique_ptr<successors>(uint_fast64_t)> maximizerSuccessors,
                std::function<std::unique_ptr<successors>(uint_fast64_t)> minimizerSuccessors,
                std::function<ValueType(uint_fast64_t, uint_fast64_t)> wMaxToMin,
                std::function<ValueType(uint_fast64_t, uint_fast64_t)> wMinToMax) const {
        }


        /*!
         * Iterators implementation
         */

        template <typename ValueType>
        TotalPayoffGame<ValueType>::successors::successors(
                uint_fast64_t state,
                const storm::storage::SparseMatrix<ValueType> &matrix,
                const storm::storage::BitVector &enabledActions)
                : state(state), matrix(matrix), enabledActions(enabledActions) {}

        template <typename ValueType>
        TotalPayoffGame<ValueType>::successors::~successors() = default;

        template <typename ValueType>
        TotalPayoffGame<ValueType>::successorsP1::successorsP1(
                uint_fast64_t state,
                const storm::storage::SparseMatrix<ValueType> &matrix,
                storm::storage::BitVector const& enabledActions)
                : successors(state, matrix, enabledActions) {}

        template <typename ValueType>
        TotalPayoffGame<ValueType>::successorsP2::successorsP2(
                uint_fast64_t action,
                const storm::storage::SparseMatrix<ValueType> &matrix,
                storm::storage::BitVector const& enabledActions)
                : successors(action, matrix, enabledActions) {}

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator TotalPayoffGame<ValueType>::successors::end() {
            return iterator(0, 0, this->enabledActions);
        }

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator TotalPayoffGame<ValueType>::successorsP1::begin() {
            uint_fast64_t end = this->state == this->matrix.getRowGroupCount() - 1 ?
                                this->matrix.getRowCount()
                                : this->matrix.getRowGroupIndices()[this->state + 1];
            return iterator(this->matrix.getRowGroupIndices()[this->state], end, this->enabledActions);
        }

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator TotalPayoffGame<ValueType>::successorsP2::begin() {
            return this->enabledActions[this->state] ?
                   iterator(this->matrix.getRow(this->state).begin(), this->matrix.getRow(this->state).end(), this->enabledActions)
                   : iterator(0, 0, this->enabledActions);
        }

        template<typename ValueType>
        TotalPayoffGame<ValueType>::iterator::iterator(uint_fast64_t rowBegin, uint_fast64_t rowEnd,
                storm::storage::BitVector const& enabledActions)
        : rowEnd(rowEnd), iterate_on_columns(false), enabledActions(enabledActions) {
            this->currentRow = this->enabledActions.getNextSetIndex(rowBegin);
        }

        template<typename ValueType>
        TotalPayoffGame<ValueType>::iterator::iterator(
                typename std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>::const_iterator entriesIteratorBegin,
                typename std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>::const_iterator entriesIteratorEnd,
                storm::storage::BitVector const& enabledActions)
                : currentRow(0), rowEnd(0),
                  matrixEntryIterator(entriesIteratorBegin),
                  ptr_end(entriesIteratorEnd),
                  iterate_on_columns(true),
                  enabledActions(enabledActions) {
            this->currentColumn = this->matrixEntryIterator->getColumn();
            this->matrixEntryIterator++;
        }

        template<typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator& TotalPayoffGame<ValueType>::iterator::operator++() {
            if (this->matrixEntryIterator == ptr_end) {
                iterate_on_columns = false;
            }
            if (this->currentRow < this->rowEnd) {
                this->currentRow = this->enabledActions.getNextSetIndex(currentRow + 1);
            }
            else if (iterate_on_columns) {
                this->currentColumn = this->matrixEntryIterator->getColumn();
                this->matrixEntryIterator++;
            }
            return *this;
        }

        template <typename ValueType>
        typename TotalPayoffGame<ValueType>::iterator TotalPayoffGame<ValueType>::iterator::operator++(int) {
            iterator it = *this;
            ++*this;
            return it;
        }

        template <typename ValueType>
        bool TotalPayoffGame<ValueType>::iterator::operator!=(const TotalPayoffGame<ValueType>::iterator &otherIterator) {
            bool stop = this->currentRow == this->rowEnd and otherIterator.currentRow == otherIterator.rowEnd
                         and (not this->iterate_on_columns) and (not otherIterator.iterate_on_columns);
            bool sameCurrentRow = not this->iterate_on_columns and not otherIterator.iterate_on_columns
                            and this->currentRow == otherIterator.currentRow;
            bool sameCurrentColumn = this->iterate_on_columns and otherIterator.iterate_on_columns
                            and this->currentColumn == otherIterator.currentColumn;
            bool isEqual = stop or sameCurrentRow or sameCurrentColumn;
            return not isEqual;
        }

        template <typename ValueType>
        uint_fast64_t TotalPayoffGame<ValueType>::iterator::operator*() {
            if (not iterate_on_columns) {
                return this->currentRow;
            }
            else {
                return this->currentColumn;
            }
        }

        template class TotalPayoffGame<double>;
        template class TotalPayoffGame<storm::RationalNumber>;

    }
}
