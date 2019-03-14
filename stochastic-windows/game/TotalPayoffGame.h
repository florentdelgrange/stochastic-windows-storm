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
             * @param mdp
             * @param rewardModelName
             * @param restrictedStateSpace
             * @param enabledActions
             */
            TotalPayoffGame(
                    storm::models::sparse::Mdp <ValueType, storm::models::sparse::StandardRewardModel<ValueType>> const &mdp,
                    std::string const &rewardModelName,
                    storm::storage::BitVector const &restrictedStateSpace,
                    storm::storage::BitVector const &enabledActions);

            std::vector<ValueType> maxTotalPayoffInf() const;
            std::vector<ValueType> minTotalPayoffSup() const;
            /**
             * Computes the set of states (P1 states -- states of the MDP, and P2 states -- actions of the MDP) from
             * which P2 can enforce a strictly negative supremum total-payoff
             */
            GameStates negSupTP() const;

            /*!
             * initialize the input BackwardTransitions structure for this MDP game
             * @param backwardTransitions an empty BackwardTransitions structure to initialize
             * @note in total-payoff games, actions may lead to states not belonging to the restricted state space
             */
            void initBackwardTransitions(BackwardTransitions &backwardTransitions) const override;

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
             * Checks if values of vectors from X are strictly positive
             */
            bool valuesStrictlyPositive(Values const& X) const;

            /*!
             * Iterators: allow to iterate on successors of states or actions, given the transition matrix, the
             * restricted state space, the set of enabled actions or a transition structure
             */

            class iterator {
            public:
                virtual ~iterator() = 0;
                virtual iterator& operator++() = 0;
                virtual bool operator!=(iterator const& otherIterator) = 0;
                virtual uint_fast64_t operator*() = 0;
                virtual uint_fast64_t operator*() const = 0;
                virtual bool end() const = 0;
            };

            class iteratorP1: public iterator {
            public:
                iteratorP1(uint_fast64_t rowBegin, uint_fast64_t rowEnd, storm::storage::BitVector const& enabledActions);
                iterator& operator++() override;
                bool operator!=(iterator const& otherIterator) override;
                uint_fast64_t operator*() override;
                uint_fast64_t operator*() const override;
                bool end() const override;
            private:
                uint_fast64_t currentRow;
                uint_fast64_t rowEnd;
                storm::storage::BitVector const& enabledActions;
            };

            class successorsIterator: public iterator {
            public:
                explicit successorsIterator(std::unique_ptr<iterator> concreteIterator);
                iterator& operator++() override;
                bool operator!=(iterator const& otherIterator) override;
                uint_fast64_t operator*() override;
                uint_fast64_t operator*() const override;
                bool end() const override;
            protected:
                std::unique_ptr<iterator> concreteIterator;
            };

            class successors {
            public:
                virtual ~successors() = 0;
                virtual successorsIterator begin() = 0;
                virtual successorsIterator end() = 0;
            };

            class successorsP1: public successors {
            public:
                successorsP1(uint_fast64_t state,
                             storm::storage::SparseMatrix<ValueType> const& matrix,
                             storm::storage::BitVector const& restrictedStateSpace,
                             storm::storage::BitVector const& enabledActions);
                successorsIterator begin() override;
                successorsIterator end() override;
            private:
                uint_fast64_t state;
                storm::storage::SparseMatrix<ValueType> const& matrix;
                storm::storage::BitVector const& restrictedStateSpace;
                storm::storage::BitVector const& enabledActions;
            };

            class forwardIteratorP2: public iterator {
            public:
                forwardIteratorP2(typename std::forward_list<uint_fast64_t>::const_iterator successorsIteratorBegin,
                                  typename std::forward_list<uint_fast64_t>::const_iterator successorsIteratorEnd);
                iterator& operator++() override;
                bool operator!=(iterator const& otherIterator) override;
                uint_fast64_t operator*() override;
                uint_fast64_t operator*() const override;
                bool end() const override;
            private:
                typename std::forward_list<uint_fast64_t>::const_iterator successorsIterator;
                typename std::forward_list<uint_fast64_t>::const_iterator ptr_end;
            };

            class forwardSuccessorsP2: public successors {
            public:
                explicit forwardSuccessorsP2(std::forward_list<uint_fast64_t> const& successorList);
                successorsIterator begin() override;
                successorsIterator end() override;
            private:
                std::forward_list<uint_fast64_t> const& successorList;
            };

            /*!
             * Compute the max total payoff inf values for all maximizer and minimizer states.
             * @note the early stopping criterion is the following: since Y is non-decreasing, as soon as the sum
             *       becomes strictly positive, it remains strictly positive for all other iterations. Thus, by enabling
             *       this early stopping criterion, you should be aware that the computation of values may stop before
             *       getting the exact values for total payoff.
             */
            Values maxTotalPayoffInf(
                    storm::Environment const& env,
                    storm::storage::BitVector const& maximizerStateSpace,
                    storm::storage::BitVector const& minimizerStateSpace,
                    std::function<std::unique_ptr<successors>(uint_fast64_t)> const& maximizerSuccessors,
                    std::function<std::unique_ptr<successors>(uint_fast64_t)> const& minimizerSuccessors,
                    std::function<ValueType(uint_fast64_t, uint_fast64_t)> const& wMaxToMin,
                    std::function<ValueType(uint_fast64_t, uint_fast64_t)> const& wMinToMax,
                    ValueType W, bool earlyStopping=false) const;
        };

    }
};



#endif //STORM_TOTALPAYOFFGAME_H
