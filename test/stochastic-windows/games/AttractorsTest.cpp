//
// Created by Florent Delgrange on 2019-08-15.
//

#include "stochastic-windows/game/TotalPayoffGame.h"
#include "gtest/gtest.h"
#include "storm-parsers/parser/PrismParser.h"
#include "storm/builder/ExplicitModelBuilder.h"
#include "storm/models/sparse/Mdp.h"

TEST(Attractors, p1AttractorsTest) {
    std::string prismModelPath = STORM_TEST_RESOURCES_DIR "/mdp/window_mp_par.nm";
    storm::prism::Program program = storm::parser::PrismParser::parse(prismModelPath);
    storm::builder::BuilderOptions options = storm::builder::BuilderOptions(true, true);
    std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(program, options).build();
    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();
    storm::storage::BitVector restrictedStateSpace(mdp->getNumberOfStates(), true);
    storm::storage::BitVector enabledActions(mdp->getNumberOfChoices(), true);
    sw::game::TotalPayoffGame<double> game(*mdp, "weights", restrictedStateSpace, enabledActions);
    sw::game::BackwardTransitions backwardTransitions;
    game.initBackwardTransitions(backwardTransitions);

    // P1 attractors
    storm::storage::BitVector T(restrictedStateSpace.size(), false);
    storm::storage::BitVector expectedSet(restrictedStateSpace.size(), false);
    T.set(0);
    expectedSet = T;
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear();
    T.set(1);
    expectedSet = T;
    expectedSet.set(0);
    expectedSet.set(1);
    expectedSet.set(3);
    expectedSet.set(6);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear();
    T.set(2);
    expectedSet = T;
    expectedSet.set(0);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear();
    T.set(3);
    expectedSet = T;
    expectedSet.set(6);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear();
    T.set(4);
    expectedSet = T;
    expectedSet.set(7);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear();
    T.set(5);
    expectedSet = T;
    expectedSet.set(8);
    expectedSet.set(10);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear();
    T.set(6);
    expectedSet = T;
    expectedSet.set(3);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear();
    T.set(7);
    expectedSet = T;
    expectedSet.set(4);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear();
    T.set(8);
    expectedSet = T;
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear();
    T.set(9);
    expectedSet = T;
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear();
    T.set(10);
    expectedSet = T;
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear();
    T.set(11);
    expectedSet = T;
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear();
    T.set(4);
    T.set(7);
    T.set(11);
    expectedSet = T;
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear();
    T.set(5);
    T.set(4);
    expectedSet = T;
    expectedSet.set(0);
    expectedSet.set(2);
    expectedSet.set(7);
    expectedSet.set(8);
    expectedSet.set(10);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    // P1 attractors of S = {7}, A = {9}
    sw::game::GameStates S;
    S.p1States = storm::storage::BitVector(restrictedStateSpace.size(), false);
    S.p2States = storm::storage::BitVector(enabledActions.size(), false);
    S.p1States.set(7, true);
    S.p2States.set(9, true);
    sw::game::GameStates A = game.attractorsP1(S, backwardTransitions);
    expectedSet = S.p1States;
    expectedSet.set(3);
    expectedSet.set(4);
    expectedSet.set(6);
    ASSERT_EQ(A.p1States, expectedSet);
    expectedSet = S.p2States;
    expectedSet.set(4);
    expectedSet.set(5);
    expectedSet.set(8);
    expectedSet.set(10);
    ASSERT_EQ(A.p2States, expectedSet);
}

TEST(Attractors, p2AttractorsTest) {
    std::string prismModelPath = STORM_TEST_RESOURCES_DIR "/mdp/window_mp_par.nm";
    storm::prism::Program program = storm::parser::PrismParser::parse(prismModelPath);
    storm::builder::BuilderOptions options = storm::builder::BuilderOptions(true, true);
    std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(program, options).build();
    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();
    storm::storage::BitVector restrictedStateSpace(mdp->getNumberOfStates(), true);
    storm::storage::BitVector enabledActions(mdp->getNumberOfChoices(), true);
    sw::game::TotalPayoffGame<double> game(*mdp, "weights", restrictedStateSpace, enabledActions);
    sw::game::BackwardTransitions backwardTransitions;
    game.initBackwardTransitions(backwardTransitions);
    sw::game::GameStates S;
    S.p1States = storm::storage::BitVector(restrictedStateSpace.size(), false);
    S.p2States = storm::storage::BitVector(enabledActions.size(), false);
    sw::game::GameStates A = game.attractorsP1(S, backwardTransitions);

    // P2 attractors
    // attractors of state 0
    S.p1States.set(0);
    storm::storage::BitVector expectedSetP1 = S.p1States;
    storm::storage::BitVector expectedSetP2 = S.p2States;
    A = game.attractorsP2(S, backwardTransitions);
    ASSERT_EQ(A.p1States, expectedSetP1);
    ASSERT_EQ(A.p2States, expectedSetP2);
    S.p1States.clear(); S.p2States.clear();
    expectedSetP1.clear(); expectedSetP2.clear();

    // attractors of state 1
    S.p1States.set(1);
    expectedSetP1.set(1);
    expectedSetP2.set(0); expectedSetP2.set(2); expectedSetP2.set(9);
    A = game.attractorsP2(S, backwardTransitions);
    ASSERT_EQ(A.p1States, expectedSetP1);
    ASSERT_EQ(A.p2States, expectedSetP2);
    S.p1States.clear(); S.p2States.clear();
    expectedSetP1.clear(); expectedSetP2.clear();

    // attractors of state 2
    S.p1States.set(2);
    expectedSetP1.set(2);
    expectedSetP2.set(1);
    A = game.attractorsP2(S, backwardTransitions);
    ASSERT_EQ(A.p1States, expectedSetP1);
    ASSERT_EQ(A.p2States, expectedSetP2);
    S.p1States.clear(); S.p2States.clear();
    expectedSetP1.clear(); expectedSetP2.clear();

    // attractors of state 3
    S.p1States.set(3);
    expectedSetP1.set(1); expectedSetP1.set(3); expectedSetP1.set(6);
    expectedSetP2.set(0); expectedSetP2.set(2); expectedSetP2.set(4); expectedSetP2.set(8); expectedSetP2.set(9);
    A = game.attractorsP2(S, backwardTransitions);
    ASSERT_EQ(A.p1States, expectedSetP1);
    ASSERT_EQ(A.p2States, expectedSetP2);
    S.p1States.clear(); S.p2States.clear();
    expectedSetP1.clear(); expectedSetP2.clear();

    // attractors of state 4
    S.p1States.set(4);
    expectedSetP1.set(2); expectedSetP1.set(4); expectedSetP1.set(7); expectedSetP1.set(11);
    expectedSetP2.set(1); expectedSetP2.set(3); expectedSetP2.set(5); expectedSetP2.set(6); expectedSetP2.set(10);
    expectedSetP2.set(11); expectedSetP2.set(15); expectedSetP2.set(16);
    A = game.attractorsP2(S, backwardTransitions);
    ASSERT_EQ(A.p1States, expectedSetP1);
    ASSERT_EQ(A.p2States, expectedSetP2);
    S.p1States.clear(); S.p2States.clear();
    expectedSetP1.clear(); expectedSetP2.clear();

    // attractors of state 5
    S.p1States.set(5);
    expectedSetP1.set(2); expectedSetP1.set(5); expectedSetP1.set(8); expectedSetP1.set(10);
    expectedSetP2.set(1); expectedSetP2.set(3); expectedSetP2.set(6); expectedSetP2.set(7); expectedSetP2.set(12);
    expectedSetP2.set(14); expectedSetP2.set(15);
    A = game.attractorsP2(S, backwardTransitions);
    ASSERT_EQ(A.p1States, expectedSetP1);
    ASSERT_EQ(A.p2States, expectedSetP2);
    S.p1States.clear(); S.p2States.clear();
    expectedSetP1.clear(); expectedSetP2.clear();

    // attractors of state 6
    S.p1States.set(6);
    expectedSetP1.set(1); expectedSetP1.set(3); expectedSetP1.set(6);
    expectedSetP2.set(0); expectedSetP2.set(2); expectedSetP2.set(4); expectedSetP2.set(8); expectedSetP2.set(9);
    A = game.attractorsP2(S, backwardTransitions);
    ASSERT_EQ(A.p1States, expectedSetP1);
    ASSERT_EQ(A.p2States, expectedSetP2);
    S.p1States.clear(); S.p2States.clear();
    expectedSetP1.clear(); expectedSetP2.clear();

    // attractors of state 7
    S.p1States.set(7);
    expectedSetP1.set(2); expectedSetP1.set(4); expectedSetP1.set(7); expectedSetP1.set(11);
    expectedSetP2.set(1); expectedSetP2.set(3); expectedSetP2.set(5); expectedSetP2.set(6); expectedSetP2.set(10);
    expectedSetP2.set(11); expectedSetP2.set(15); expectedSetP2.set(16);
    A = game.attractorsP2(S, backwardTransitions);
    ASSERT_EQ(A.p1States, expectedSetP1);
    ASSERT_EQ(A.p2States, expectedSetP2);
    S.p1States.clear(); S.p2States.clear();
    expectedSetP1.clear(); expectedSetP2.clear();

    // attractors of state 8
    S.p1States.set(8);
    expectedSetP1.set(2); expectedSetP1.set(5); expectedSetP1.set(8); expectedSetP1.set(10);
    expectedSetP2.set(1); expectedSetP2.set(3); expectedSetP2.set(6); expectedSetP2.set(7); expectedSetP2.set(12);
    expectedSetP2.set(14); expectedSetP2.set(15);
    A = game.attractorsP2(S, backwardTransitions);
    ASSERT_EQ(A.p1States, expectedSetP1);
    ASSERT_EQ(A.p2States, expectedSetP2);
    S.p1States.clear(); S.p2States.clear();
    expectedSetP1.clear(); expectedSetP2.clear();

    // attractors of state 8
    S.p1States.set(8);
    expectedSetP1.set(2); expectedSetP1.set(5); expectedSetP1.set(8); expectedSetP1.set(10);
    expectedSetP2.set(1); expectedSetP2.set(3); expectedSetP2.set(6); expectedSetP2.set(7); expectedSetP2.set(12);
    expectedSetP2.set(14); expectedSetP2.set(15);
    A = game.attractorsP2(S, backwardTransitions);
    ASSERT_EQ(A.p1States, expectedSetP1);
    ASSERT_EQ(A.p2States, expectedSetP2);
    S.p1States.clear(); S.p2States.clear();
    expectedSetP1.clear(); expectedSetP2.clear();

    // attractors of state 9
    S.p1States.set(9);
    expectedSetP1.set(9);
    expectedSetP2.set(6); expectedSetP2.set(13); expectedSetP2.set(15);
    A = game.attractorsP2(S, backwardTransitions);
    ASSERT_EQ(A.p1States, expectedSetP1);
    ASSERT_EQ(A.p2States, expectedSetP2);
    S.p1States.clear(); S.p2States.clear();
    expectedSetP1.clear(); expectedSetP2.clear();

    // attractors of state 10
    S.p1States.set(10);
    expectedSetP1.set(10);
    expectedSetP2.set(7);
    A = game.attractorsP2(S, backwardTransitions);
    ASSERT_EQ(A.p1States, expectedSetP1);
    ASSERT_EQ(A.p2States, expectedSetP2);
    S.p1States.clear(); S.p2States.clear();
    expectedSetP1.clear(); expectedSetP2.clear();

    // attractors of state 11
    S.p1States.set(11);
    expectedSetP1.set(11);
    expectedSetP2.set(11); expectedSetP2.set(16);
    A = game.attractorsP2(S, backwardTransitions);
    ASSERT_EQ(A.p1States, expectedSetP1);
    ASSERT_EQ(A.p2States, expectedSetP2);
    S.p1States.clear(); S.p2States.clear();
    expectedSetP1.clear(); expectedSetP2.clear();

    // attractors of set S = {4, 10}, A = {0}
    S.p1States.set(4); S.p1States.set(10); S.p2States.set(0);
    expectedSetP1.set(0); expectedSetP1.set(2); expectedSetP1.set(4); expectedSetP1.set(5); expectedSetP1.set(7);
    expectedSetP1.set(8); expectedSetP1.set(10); expectedSetP1.set(11);
    expectedSetP2.set(0); expectedSetP2.set(1); expectedSetP2.set(3); expectedSetP2.set(5); expectedSetP2.set(6);
    expectedSetP2.set(7); expectedSetP2.set(10); expectedSetP2.set(11); expectedSetP2.set(12);
    expectedSetP2.set(14); expectedSetP2.set(15); expectedSetP2.set(16);
    A = game.attractorsP2(S, backwardTransitions);
    ASSERT_EQ(A.p1States, expectedSetP1);
    ASSERT_EQ(A.p2States, expectedSetP2);
    S.p1States.clear(); S.p2States.clear();
    expectedSetP1.clear(); expectedSetP2.clear();
}