//
// Created by Florent Delgrange on 2019-01-29.
//

#ifndef STORM_PREDECESSORSSQUAREDLINKEDLIST_H
#define STORM_PREDECESSORSSQUAREDLINKEDLIST_H

class Node;

#include <memory>
#include <vector>
#include <storm/storage/SparseMatrix.h>
#include <storm/storage/BitVector.h>

namespace sw {
    namespace storage {
        /*!
         * The predecessor squared linked list is a dynamic triple linked list data structure where each node
         * represents a relation between a state and an action, allowing to iterate on each successor-state of
         * an action and each predecessor-action of a state (i.e., actions leading to this state).
         * Moreover, it allows to dynamically disable actions and to check if it remains enabled actions in a given state s.
         * By disabling an enabled action a of a state s, for each successor s' of a, a will not be considered anymore
         * as a predecessor of s' and as an enabled action of s.
         * @tparam ValueType
         */
        template <typename ValueType>
        class PredecessorsSquaredLinkedList {
        public:

            class Node {
            public:
                std::shared_ptr<Node> prevAction, nextAction, nextState;
                uint_fast64_t actionIndex;
            };

            class MainNode: public Node {
            public:
                std::shared_ptr<Node> lastNode;
            };

            class iterator {
            public:
                explicit iterator(std::shared_ptr<Node> node);
                iterator& operator=(std::shared_ptr<Node> node);
                // prefix ++
                iterator& operator++();
                // postfix ++
                iterator operator++(int);
                bool operator!=(iterator const& otherIterator);
                uint_fast64_t operator*();
            private:
                std::shared_ptr<Node> currentNode;
            };

            class actions_list {
            public:
                actions_list(std::shared_ptr<Node> firstNode);
                iterator begin();
                iterator end();
            private:
                std::shared_ptr<Node> firstNode;
            };

            PredecessorsSquaredLinkedList(
                    storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                    storm::storage::BitVector const& restrictedStateSpace,
                    storm::storage::BitVector const& enabledActions);

            /*!
             * gives the state-predecessor of the input action, i.e., the state for which the input action is enabled.
             */
            uint_fast64_t getActionPredecessor(uint_fast64_t action);
            /*!
             * disable the input action
             */
            void disableAction(uint_fast64_t action);
            /*!
             * checks if it remains an enabled action for the input state
             */
            bool hasEnabledActions(uint_fast64_t state);

            /*!
             * gives the action-predecessors of the input state, i.e., the actions leading to the input state.
             * @param state
             * @return an object allowing to iterate on actions leading to the input state
             */
            actions_list getStatePredecessors(uint_fast64_t state);
            actions_list getStatePredecessors(uint_fast64_t state) const;

            friend std::ostream& operator<<(std::ostream& out, PredecessorsSquaredLinkedList<ValueType> const& predList) {
                for (uint_fast64_t state = 0; state < predList.states.size(); ++ state) {
                    if (predList.states[state]->nextAction != nullptr) {
                        out << "[" << state << "] -> ";
                        for (uint_fast64_t action: predList.getStatePredecessors(state)) {
                            out << action << " -> ";
                        }
                        out << "*" << std::endl;
                    }
                }
                return out;
            }

        private:
            std::vector<std::shared_ptr<MainNode>> states;
            std::vector<std::shared_ptr<MainNode>> actions;
            std::vector<uint_fast64_t> actionPredecessors;
            std::vector<uint_fast64_t> numberOfEnabledActions;
        };
    }
}


#endif //STORM_PREDECESSORSSQUAREDLINKEDLIST_H
