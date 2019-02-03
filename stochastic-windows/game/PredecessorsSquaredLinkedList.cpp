//
// Created by Florent Delgrange on 2019-01-29.
//

#include "PredecessorsSquaredLinkedList.h"

namespace sw {
    namespace storage {

        template<typename ValueType>
        PredecessorsSquaredLinkedList<ValueType>::PredecessorsSquaredLinkedList(
                storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                storm::storage::BitVector const &restrictedStateSpace,
                storm::storage::BitVector const &enabledActions)
                : states(transitionMatrix.getRowGroupCount()),
                  actions(transitionMatrix.getRowCount()),
                  actionPredecessors(transitionMatrix.getRowCount()),
                  numberOfEnabledActions(transitionMatrix.getRowGroupCount(), 0) {

            for (uint_fast64_t const& state: restrictedStateSpace) {
                this->states[state] = std::shared_ptr<MainNode>(new MainNode());
            }
            for (uint_fast64_t state: restrictedStateSpace) {
                for (uint_fast64_t action = enabledActions.getNextSetIndex(transitionMatrix.getRowGroupIndices()[state]);
                     action < transitionMatrix.getRowGroupIndices()[state + 1];
                     action = enabledActions.getNextSetIndex(action + 1)) {
                    this->actionPredecessors[action] = state;
                    this->actions[action] = std::shared_ptr<MainNode>(new MainNode());
                    this->numberOfEnabledActions[state] += 1;
                    for (const auto &entry: transitionMatrix.getRow(action)) {
                        uint_fast64_t successorState = entry.getColumn();
                        std::shared_ptr<Node> node(new Node());
                        node->actionIndex = action;
                        std::shared_ptr<Node> lastStateNode;
                        std::shared_ptr<Node> lastActionNode;
                        if (this->states[successorState]->lastNode == nullptr) {
                            lastStateNode = this->states[successorState];
                        }
                        else {
                            lastStateNode = this->states[successorState]->lastNode;
                        }
                        if (this->actions[action]->lastNode == nullptr) {
                            lastActionNode = this->actions[action];
                        }
                        else {
                            lastActionNode = this->actions[action]->lastNode;
                        }
                        node->prevAction = lastStateNode;
                        lastStateNode->nextAction = node;
                        lastActionNode->nextState = node;
                        this->states[successorState]->lastNode = node;
                        this->actions[action]->lastNode = node;
                    }
                }
            }
        }

        template<typename ValueType>
        void PredecessorsSquaredLinkedList<ValueType>::disableAction(uint_fast64_t action) {
            assert(this->actions.size() > action);
            while (this->actions[action]->nextState != nullptr) {
                std::shared_ptr<Node> currentNode = this->actions[action]->nextState;
                std::shared_ptr<Node> leftNode = currentNode->prevAction;
                leftNode->nextAction = currentNode->nextAction;
                if (leftNode->nextAction != nullptr) {
                    leftNode->nextAction->prevAction = leftNode;
                }
                this->actions[action]->nextState = currentNode->nextState;
            }
            this->actions[action]->lastNode = nullptr;
            this->numberOfEnabledActions[this->getActionPredecessor(action)] -= 1;
        }

        template<typename ValueType>
        uint_fast64_t PredecessorsSquaredLinkedList<ValueType>::getActionPredecessor(uint_fast64_t action) {
            assert(this->actions.size() > action);
            return this->actionPredecessors[action];
        }

        template<typename ValueType>
        PredecessorsSquaredLinkedList<ValueType>::iterator::iterator(std::shared_ptr<Node> node)
        : currentNode(node) {}

        template<typename ValueType>
        typename PredecessorsSquaredLinkedList<ValueType>::iterator& PredecessorsSquaredLinkedList<ValueType>::iterator::operator++() {
           if (this->currentNode != nullptr) {
               this->currentNode = this->currentNode->nextAction;
           }
           return *this;
        }

        template<typename ValueType>
        typename PredecessorsSquaredLinkedList<ValueType>::iterator PredecessorsSquaredLinkedList<ValueType>::iterator::operator++(int) {
            iterator it = *this;
            ++*this;
            return it;
        }

        template<typename ValueType>
        bool PredecessorsSquaredLinkedList<ValueType>::iterator::operator!=(
                const PredecessorsSquaredLinkedList<ValueType>::iterator &otherIterator) {
            return this->currentNode != otherIterator.currentNode;
        }

        template<typename ValueType>
        uint_fast64_t PredecessorsSquaredLinkedList<ValueType>::iterator::operator*() {
            return this->currentNode->actionIndex;
        }

        template<typename ValueType>
        typename PredecessorsSquaredLinkedList<ValueType>::iterator&
        PredecessorsSquaredLinkedList<ValueType>::iterator::operator=(std::shared_ptr<Node> node) {
            this->currentNode = node;
            return *this;
        }

        template<typename ValueType>
        PredecessorsSquaredLinkedList<ValueType>::actions_list::actions_list(std::shared_ptr<Node> firstNode)
        : firstNode(firstNode) {}

        template<typename ValueType>
        typename PredecessorsSquaredLinkedList<ValueType>::iterator PredecessorsSquaredLinkedList<ValueType>::actions_list::begin() {
            return PredecessorsSquaredLinkedList<ValueType>::iterator(firstNode);
        }

        template<typename ValueType>
        typename PredecessorsSquaredLinkedList<ValueType>::iterator PredecessorsSquaredLinkedList<ValueType>::actions_list::end() {
            return PredecessorsSquaredLinkedList<ValueType>::iterator(nullptr);
        }

        template<typename ValueType>
        typename PredecessorsSquaredLinkedList<ValueType>::actions_list PredecessorsSquaredLinkedList<ValueType>::getStatePredecessors(
                uint_fast64_t state) {
            assert(this->states.size() > state);
            if (this->states[state] != nullptr) {
                return actions_list(this->states[state]->nextAction);
            }
            else {
                return actions_list(this->states[state]);
            }
        }

        template<typename ValueType>
        typename PredecessorsSquaredLinkedList<ValueType>::actions_list PredecessorsSquaredLinkedList<ValueType>::getStatePredecessors(
                uint_fast64_t state) const {
            assert(this->states.size() > state);
            if (this->states[state] != nullptr) {
                return actions_list(this->states[state]->nextAction);
            }
            else {
                return actions_list(this->states[state]);
            }
        }

        template<typename ValueType>
        bool PredecessorsSquaredLinkedList<ValueType>::hasEnabledActions(uint_fast64_t state) {
            assert(this->states.size() > state);
            return this->numberOfEnabledActions[state] > 0;
        }

        template class PredecessorsSquaredLinkedList<double>;
        template class PredecessorsSquaredLinkedList<storm::RationalNumber>;

    }
}