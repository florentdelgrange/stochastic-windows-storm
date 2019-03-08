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
#include <storm/utility/vector.h>
#include <storm/environment/Environment.h>
#include "storm/environment/solver/MinMaxSolverEnvironment.h"

namespace sw {
    namespace game {

        template <typename ValueType>
        class TotalPayoffGame: public MdpGame<ValueType> {
        public:
            /*!
             * The implementations of algorithms of this class for the particular case of MDPs as games are the ones from
             * Pseudopolynomial iterative algorithm to solve total-payoff games and min-cost reachability games
             * Brihaye, T., Geeraerts, G., Haddad, A. et al. Acta Informatica (2017) 54: 85.
             * https://doi.org/10.1007/s00236-016-0276-z
             * arXiv:1407.5030v4
             */
            TotalPayoffGame(storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                            std::string const &rewardModelName,
                            storm::storage::BitVector const &restrictedStateSpace,
                            storm::storage::BitVector const &enabledActions,
                            bool initTransitionStructure=false);

            std::vector<ValueType> maxTotalPayoffInf() const;
            std::vector<ValueType> minTotalPayoffSup() const;
            /**
             * Computes the set of states from which P2 can enforce a strictly negative supremum total-payoff
             */
            GameStates negSupTP() const;

            /*!
             * initialize the input BackwardTransitions structure for this MDP game
             * @param backwardTransitions an empty BackwardTransitions structure to initialize
             * @note in total-payoff games, actions may lead to states not belonging to the restricted state space
             */
            void initBackwardTransitions(BackwardTransitions &backwardTransitions) const override;

            BackwardTransitions const& getBackwardTransition();

            /*!
             * Iterators: allow to iterate on successors of states or actions, given the transition matrix, the
             * restricted state space, the set of enabled actions or a transition structure
             */

            class iterator {
            public:
                iterator(uint_fast64_t rowBegin, uint_fast64_t rowEnd,
                         storm::storage::BitVector const& restrictedStateSpace,
                         storm::storage::BitVector const& enabledActions);
                iterator(typename std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>::const_iterator entriesIteratorBegin,
                         typename std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>::const_iterator entriesIteratorEnd,
                         storm::storage::BitVector const& restrictedStateSpace,
                         storm::storage::BitVector const& enabledActions);
                iterator(typename std::forward_list<uint_fast64_t>::const_iterator successorsIteratorBegin,
                         typename std::forward_list<uint_fast64_t>::const_iterator successorsIteratorEnd,
                         storm::storage::BitVector const& restrictedStateSpace,
                         storm::storage::BitVector const& enabledActions);
                // prefix ++
                iterator& operator++();
                // postfix ++
                const iterator operator++(int);
                bool operator!=(iterator const& otherIterator);
                uint_fast64_t operator*();
            private:
                uint_fast64_t currentRow;
                uint_fast64_t rowEnd;
                uint_fast64_t currentColumn;
                typename std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>::const_iterator matrixEntryIterator;
                typename std::vector<storm::storage::MatrixEntry<uint_fast64_t, ValueType>>::const_iterator matrix_ptr_end;
                typename std::forward_list<uint_fast64_t>::const_iterator successorsIterator;
                typename std::forward_list<uint_fast64_t>::const_iterator successors_ptr_end;
                bool iterate_on_columns;
                bool forwardSuccessors;
                storm::storage::BitVector const& restrictedStateSpace;
                storm::storage::BitVector const& enabledActions;
            };

            class successors {

            public:
                successors(uint_fast64_t state,
                           storm::storage::SparseMatrix<ValueType> const& matrix,
                           storm::storage::BitVector const& restrictedStateSpace,
                           storm::storage::BitVector const& enabledActions);
                virtual ~successors() = 0;
                virtual iterator begin() = 0;
                virtual iterator end();

            protected:
                uint_fast64_t state;
                storm::storage::SparseMatrix<ValueType> const& matrix;
                storm::storage::BitVector const& restrictedStateSpace;
                storm::storage::BitVector const& enabledActions;
            };

            class successorsP1: public successors {

            public:
                successorsP1(uint_fast64_t state,
                             storm::storage::SparseMatrix<ValueType> const& matrix,
                             storm::storage::BitVector const& restrictedStateSpace,
                             storm::storage::BitVector const& enabledActions);
                iterator begin() override;

            };

            class successorsP2: public successors {
            public:
                successorsP2(uint_fast64_t state,
                             storm::storage::SparseMatrix<ValueType> const& matrix,
                             storm::storage::BitVector const& restrictedStateSpace,
                             storm::storage::BitVector const& enabledActions);
                iterator begin() override;
            };

        private:
            /**
             * Name of the reward model to consider
             */
            std::string const& rewardModelName;
            /**
             * Reward Model to consider
             */
            storm::models::sparse::StandardRewardModel<ValueType> const& rewardModel;

            struct Values {
                std::vector<ValueType> max;
                std::vector<ValueType> min;
            };

            struct ForwardTransitions {
                std::vector<std::forward_list<uint_fast64_t>> successors;
            };

            BackwardTransitions backwardTransitions;
            ForwardTransitions forwardTransitions;

            /*!
             * Compares the given elements and determines whether they are equal modulo the given precision. The provided flag
             * additionally specifies whether the error is computed in relative or absolute terms.
             *
             * @param val1 The first value to compare.
             * @param val2 The second value to compare.
             * @param precision The precision up to which the elements are compared.
             * @param relativeError the error is computed relative to the second value.
             * @return True iff the elements are considered equal.
             * @note same than the version of storm::utility::vector but with infinite values handling
             */
            bool equalModuloPrecision(ValueType const& val1, ValueType const& val2,
                                      ValueType const& precision, bool relativeError) const;

            /*!
             * Compares the two vectors and determines whether they are equal modulo the provided precision. Depending on whether the
             * flag is set, the difference between the vectors is computed relative to the value or in absolute terms.
             * @note same than storm::utility::vector::equalModuloPrecision but with infinite values handling
             */
            bool vectorEquality(std::vector<ValueType> const& vectorLeft,
                                std::vector<ValueType> const& vectorRight,
                                ValueType const& precision, bool relativeError) const;

            /*!
             * Checks if values of vectors from X are equal to values of vectors from Y
             */
            bool valuesEqual(Values const& X, Values const& Y, ValueType precision, bool relativeError) const;

            /*!
             * Compute the max total payoff inf values for all maximizer and minimizer states.
             */
            Values maxTotalPayoffInf(
                    storm::Environment const& env,
                    storm::storage::BitVector const& maximizerStateSpace,
                    storm::storage::BitVector const& minimizerStateSpace,
                    std::function<std::unique_ptr<successors>(uint_fast64_t)> const& maximizerSuccessors,
                    std::function<std::unique_ptr<successors>(uint_fast64_t)> const& minimizerSuccessors,
                    std::function<ValueType(uint_fast64_t, uint_fast64_t)> const& wMaxToMin,
                    std::function<ValueType(uint_fast64_t, uint_fast64_t)> const& wMinToMax,
                    ValueType W) const;

            class forwardSuccessorsP2: public successors {
            public:
                forwardSuccessorsP2(
                        uint_fast64_t action,
                        storm::storage::SparseMatrix<ValueType> const& matrix,
                        ForwardTransitions const& forwardTransitions,
                        storm::storage::BitVector const& restrictedStateSpace,
                        storm::storage::BitVector const& enabledActions);
                iterator begin() override;
            protected:
                std::forward_list<uint_fast64_t> const& successorList;
            };

        };


    }
};



#endif //STORM_TOTALPAYOFFGAME_H
