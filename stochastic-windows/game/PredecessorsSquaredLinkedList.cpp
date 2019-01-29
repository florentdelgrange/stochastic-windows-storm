//
// Created by Florent Delgrange on 2019-01-29.
//

#include "PredecessorsSquaredLinkedList.h"

namespace sw {
    namespace Game {
        namespace storage {

            template<typename ValueType>
            PredecessorsSquaredLinkedList::PredecessorsSquaredLinkedList(
                    storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                    storm::storage::BitVector const &restrictedStateSpace,
                    storm::storage::BitVector const &enabledActions)
                    : states(transitionMatrix.getRowGroupCount()),
                      actions(transitionMatrix.getRowCount()),
                      stateOfAction(transitionMatrix.getRowCount()) {

                for (uint_fast64_t state: restrictedStateSpace) {
                    for (uint_fast64_t action = enabledActions.getNextSetIndex(transitionMatrix.getRowGroupIndices()[state]);
                         action < transitionMatrix.getRowGroupIndices()[state + 1];
                         action = enabledActions.getNextSetIndex(action + 1)) {
                        this->stateOfAction[action] = state;
                        for (auto entry: transitionMatrix.getRow(action)) {
                            uint_fast64_t successorState = entry.getColumn();
                            std::shared_ptr<Node> node(new Node());
                            node->actionIndex = action;
                            if (this->states[successorState]->nextAction == nullptr) {
                                this->states[successorState]->nextAction = node;
                            }
                            if (this->actions[action]->nextAction == nullptr) {
                                this->actions[action]->nextAction = node;
                            }
                            node->prevAction = this->states[successorState]->lastNode;
                            this->states[successorState]->lastNode->nextAction = node;
                            this->actions[action]->lastNode->nextState = node;
                            this->states[successorState]->lastNode = node;
                            this->actions[action]->lastNode = node;
                        }
                    }
                }
            }


            void PredecessorsSquaredLinkedList::disableAction(uint_fast64_t action) {

                while (this->actions[action]->nextState != nullptr) {
                    std::shared_ptr<Node> currentAction = this->actions[action]->nextState;
                    std::shared_ptr<Node> previousAction = currentAction->prevAction;
                    previousAction->nextAction = currentAction->nextAction;
                    if (previousAction->nextAction != nullptr) {
                        previousAction->nextAction->prevAction = previousAction;
                    }
                    this->actions[action]->nextState = currentAction->nextState;
                    currentAction->remove();
                }
                this->actions[action]->lastNode = nullptr;
            }

            uint_fast64_t PredecessorsSquaredLinkedList::getStateOfAction(uint_fast64_t action) {
                return this->stateOfAction[action];
            }

        }
    }
}