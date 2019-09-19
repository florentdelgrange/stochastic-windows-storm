//
// Created by Florent Delgrange on 2019-08-15.
//

#include "gtest/gtest.h"
#include "storm-parsers/parser/PrismParser.h"
#include "storm/builder/ExplicitModelBuilder.h"
#include "storm/models/sparse/Mdp.h"
#include "stochastic-windows/game/TotalPayoffGame.h"

TEST(TotalPayoffGames, valueIterationTest) {
    const double precision = 1e-5;
    // model initialization
    std::string prismModelPath = STORM_TEST_RESOURCES_DIR "/mdp/window_mp_par.nm";
    storm::prism::Program program = storm::parser::PrismParser::parse(prismModelPath);
    storm::builder::BuilderOptions options = storm::builder::BuilderOptions(true, true);
    std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(program, options).build();
    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();
    // total payoff game initialization
    storm::storage::BitVector restrictedStateSpace(mdp->getNumberOfStates(), true);
    storm::storage::BitVector enabledActions(mdp->getNumberOfChoices(), true);
    sw::game::TotalPayoffGame<double> game(*mdp, "weights", restrictedStateSpace, enabledActions);
    // Max total payoff inf values
    std::vector<double> values = game.maxTotalPayoffInf();
    ASSERT_EQ(-1 * storm::utility::infinity<double>(), values[0]);
    ASSERT_EQ(-1 * storm::utility::infinity<double>(), values[1]);
    ASSERT_EQ(-1 * storm::utility::infinity<double>(), values[2]);
    ASSERT_EQ(storm::utility::infinity<double>(), values[3]);
    EXPECT_NEAR(-5, values[4], precision);
    ASSERT_EQ(-1 * storm::utility::infinity<double>(), values[5]);
    ASSERT_EQ(storm::utility::infinity<double>(), values[6]);
    EXPECT_NEAR(0, values[7], precision);
    ASSERT_EQ(-1 * storm::utility::infinity<double>(), values[8]);
    ASSERT_EQ(-1 * storm::utility::infinity<double>(), values[9]);
    ASSERT_EQ(-1 * storm::utility::infinity<double>(), values[10]);
    EXPECT_NEAR(-3, values[11], precision);
    // Min total payoff sup values
    values = game.minTotalPayoffSup();
    ASSERT_EQ(-1 * storm::utility::infinity<double>(), values[0]);
    ASSERT_EQ(-1 * storm::utility::infinity<double>(), values[1]);
    ASSERT_EQ(-1 * storm::utility::infinity<double>(), values[2]);
    ASSERT_EQ(storm::utility::infinity<double>(), values[3]);
    EXPECT_NEAR(0, values[4], precision);
    ASSERT_EQ(-1 * storm::utility::infinity<double>(), values[5]);
    ASSERT_EQ(storm::utility::infinity<double>(), values[6]);
    EXPECT_NEAR(5, values[7], precision);
    ASSERT_EQ(-1 * storm::utility::infinity<double>(), values[8]);
    ASSERT_EQ(-1 * storm::utility::infinity<double>(), values[9]);
    ASSERT_EQ(-1 * storm::utility::infinity<double>(), values[10]);
    EXPECT_NEAR(2, values[11], precision);
}
