//
// Created by Florent Delgrange on 2019-08-15.
//

#include <stochastic-windows/game/TotalPayoffGame.h>
#include "gtest/gtest.h"
#include "storm-parsers/parser/PrismParser.h"
#include "storm/builder/ExplicitModelBuilder.h"
#include "storm/models/sparse/Mdp.h"

TEST(Attractors, attractorTest) {
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

    // Singleton P1 attractors
    storm::storage::BitVector T(restrictedStateSpace.size(), false);
    storm::storage::BitVector expectedSet(restrictedStateSpace.size(), false);
    T.set(0);
    expectedSet = T;
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear(); T.set(1);
    expectedSet = T; expectedSet.set(0); expectedSet.set(1); expectedSet.set(3); expectedSet.set(6);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear(); T.set(2);
    expectedSet = T; expectedSet.set(0);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear(); T.set(3);
    expectedSet = T; expectedSet.set(6);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear(); T.set(4);
    expectedSet = T; expectedSet.set(7);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear(); T.set(5);
    expectedSet = T; expectedSet.set(8); expectedSet.set(10);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear(); T.set(6);
    expectedSet = T; expectedSet.set(3);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear(); T.set(7);
    expectedSet = T; expectedSet.set(4);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear(); T.set(8);
    expectedSet = T;
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear(); T.set(9);
    expectedSet = T;
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear(); T.set(10);
    expectedSet = T;
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear(); T.set(11);
    expectedSet = T;
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear(); T.set(4); T.set(7); T.set(11);
    expectedSet = T;
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    T.clear(); T.set(5); T.set(4);
    expectedSet = T; expectedSet.set(0); expectedSet.set(2); expectedSet.set(7); expectedSet.set(8); expectedSet.set(10);
    ASSERT_EQ(expectedSet, game.attractorsP1(T, backwardTransitions));
    sw::game::GameStates S;
    S.p1States = storm::storage::BitVector(restrictedStateSpace.size(), false);
    S.p2States = storm::storage::BitVector(enabledActions.size(), false);
    S.p1States.set(7, true); S.p2States.set(9, true);
    sw::game::GameStates A = game.attractorsP1(S, backwardTransitions);
    expectedSet = S.p1States; expectedSet.set(3); expectedSet.set(4); expectedSet.set(6);
    ASSERT_EQ(A.p1States, expectedSet);
    expectedSet = S.p2States; expectedSet.set(4); expectedSet.set(5); expectedSet.set(8); expectedSet.set(10);
    ASSERT_EQ(A.p2States, expectedSet);
}