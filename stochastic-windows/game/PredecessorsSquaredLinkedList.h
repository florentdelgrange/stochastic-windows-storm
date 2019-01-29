//
// Created by Florent Delgrange on 2019-01-29.
//

#ifndef STORM_PREDECESSORSSQUAREDLINKEDLIST_H
#define STORM_PREDECESSORSSQUAREDLINKEDLIST_H

#include <memory>
#include <vector>
#include <storm/storage/SparseMatrix.h>
#include <storm/storage/BitVector.h>

namespace sw {
    namespace Game {
        namespace storage {

            class PredecessorsSquaredLinkedList {
            public:
                template <typename ValueType>
                PredecessorsSquaredLinkedList(
                        storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                        storm::storage::BitVector const& restrictedStateSpace,
                        storm::storage::BitVector const& enabledActions);

                uint_fast64_t getStateOfAction(uint_fast64_t action);
                void disableAction(uint_fast64_t action);

            private:
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
                std::vector<std::shared_ptr<MainNode>> states;
                std::vector<std::shared_ptr<MainNode>> actions;
                std::vector<uint_fast64_t> stateOfAction;
            };
        }
    }
}


#endif //STORM_PREDECESSORSSQUAREDLINKEDLIST_H
