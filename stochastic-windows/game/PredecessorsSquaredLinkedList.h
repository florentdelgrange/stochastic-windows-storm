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
    namespace Game {
        namespace storage {

            template <typename ValueType>
            class PredecessorsSquaredLinkedList {
            public:

                class Node {
                public:
                    ~Node() {};
                    std::shared_ptr<Node> prevAction, nextAction, nextState;
                    uint_fast64_t actionIndex;
                    void remove() { delete this; }
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
                    //postfix ++
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

                uint_fast64_t getActionPredecessor(uint_fast64_t action);

                void disableAction(uint_fast64_t action);

                actions_list getStatePredecessors(uint_fast64_t state) const;
                actions_list getStatePredecessors(uint_fast64_t state);

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
            };
        }
    }
}


#endif //STORM_PREDECESSORSSQUAREDLINKEDLIST_H
