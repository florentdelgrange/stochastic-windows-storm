//
// Created by Florent Delgrange on 09/11/2018.
//

#include <storm/storage/SparseMatrix.h>
#include <storm/models/sparse/Model.h>
#include <iostream>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/MaximalEndComponentDecomposition.h>

#ifndef STORM_BNDGOODWINDOWMP_H
#define STORM_BNDGOODWINDOWMP_H

#endif //STORM_BNDGOODWINDOWMP_H

namespace sw {

    template <typename ValueType>

    class BndGoodWindowMP {
    public:

        struct ECsUnfolding {
            // For each MEC in the MEC decomposition, create a matrix representing the unfolded MEC for the window MP
            // objective.
            std::vector<storm::storage::SparseMatrix<ValueType>> matrices;
            // Arrays mapping, for each MEC of the original MDP, each row of the new matrix
            // (i.e., each transition of the unfolding of this MEC) to a row in the original matrix
            // (i.e., the original MDP).
            // For the sink rows added to EC states, an arbitrary row of the original matrix that stays inside the EC
            // is given.
            std::vector<std::vector<uint_fast64_t>> newToOldRowMapping;
            // Array mapping for each state of the original MDP a tuple containing, at the first position, the index of
            // the new matrix containing it (i.e., the index of the corresponding MEC), and, at the second position,
            // an array of size l_max containing in each position l a hash table mapping a (current negative) weight w
            // to an index in the new matrix corresponding to the the following tuple :
            // (s, w, l), where s is the original state of the MDP, w is the current value of the window (sum of
            // weights), and l is the current number of steps of the window.
            std::vector<std::tuple<uint_fast64_t, std::vector<std::map<ValueType, uint_fast64_t>>>> oldToNewStateMapping;
        };

        /**
         * Unfold the end components of the mdp entered in parameter for the window mean payoff problem w.r.t. the
         * maximal windows' length l_max.
         *
         * @param mdp Markov decision process for which the ECs will be unfolded
         * @param rewardModelName name of the reward model following which the mdp will be unfolded
         * @param l_max the maximum length of windows to consider
         * @return
         */
        static ECsUnfolding unfoldECs(std::shared_ptr<storm::models::sparse::Mdp<ValueType>> mdp,
                std::string const& rewardModelName,
                uint_fast64_t l_max) {
            storm::storage::SparseMatrix<ValueType> originalMatrix = mdp->getTransitionMatrix();
            assert(mdp->hasRewardModel(rewardModelName));
            storm::models::sparse::StandardRewardModel<ValueType> rewardModel = mdp->getRewardModel(rewardModelName);
            assert(rewardModel.hasStateActionRewards());
            storm::storage::MaximalEndComponentDecomposition<ValueType> mecDecomposition(*mdp);

            // Initialization phase
            // Array containing the sparse matrix for each MEC.
            ECsUnfolding result;
            result.matrices = std::vector<storm::storage::SparseMatrix<ValueType>>(mecDecomposition.size());
            // Matrix containing all data about the unfolding
            result.oldToNewStateMapping = std::vector<std::tuple<uint_fast64_t, std::vector<std::map<ValueType, uint_fast64_t>>>>(mdp->getNumberOfStates());
            storm::storage::MaximalEndComponent mec;
            for (int k = 0; k < mecDecomposition.size(); ++k){
                 mec = mecDecomposition[k];
                 storm::storage::SparseMatrix<ValueType> matrixBuilder = storm::storage::SparseMatrixBuilder<ValueType>::SparseMatrixBuilder();
                 for (auto state : mec.getStateSet()){
                     // Initialize a tuple with the mec containing state and an empty vector of size l_max
                     result.oldToNewStateMapping[state] = std::make_tuple(k, std::vector<std::map<ValueType, uint_fast64_t>>(l_max + 1));
                 }
                 // Unfold the kth MEC
                 for (auto state: mec.getStateSet()){
                     unfoldFrom(state, 0., 0, result);
                 }
                result.matrices[k] = matrixBuilder.build();
            }
            return result;
        };

    private:
        /**
         * Unfold a MEC from a given state. The result fields oldToNewStateMapping and newToOldRowMapping are filled
         * accordingly.
         *
         * @param state
         * @param currentSumOfWeights
         * @param l
         * @param result
         */
        static void unfoldFrom(u_int64_t state, ValueType const &currentSumOfWeights, uint_fast64_t l,
                    // storm::storage::SparseMatrixBuilder<ValueType> const &matrixBuilder,
                    ECsUnfolding const &result){
            // kth MEC containing state
            u_int64_t k;
            // window vector of size l_max
            std::vector<std::map<ValueType, uint_fast64_t>> windowVector;
            std::tie(k, windowVector) = result.oldToNewStateMapping[state];
            std::map<ValueType, uint_fast64_t> weightsMap = windowVector[l];
            auto i = weightsMap.find(currentSumOfWeights);
            if (i == weightsMap.end()){
                // If the state (state, currentSomOfWeights, l) does not yet exists, fill the newToOldRowMapping consequently
                i = result.newToOldRowMapping.size();
                result.newToOldRowMapping.push_back(std::vector<uint_fast64_t >());
                weightsMap[currentSumOfWeights] = i;
            }
            
        }
    };

}