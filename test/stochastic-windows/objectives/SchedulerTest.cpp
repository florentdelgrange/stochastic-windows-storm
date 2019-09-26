//
// Created by florent on 23/09/19.
//

#include <storm/models/sparse/Mdp.h>
#include <storm/settings/modules/MinMaxEquationSolverSettings.h>
#include <storm-parsers/parser/PrismParser.h>
#include "gtest/gtest.h"
#include "storm/builder/ExplicitModelBuilder.h"
#include "storm-parsers/parser/NondeterministicModelParser.h"
#include "stochastic-windows/util/Graphviz.h"
#include "stochastic-windows/fixedwindow/FixedWindowObjective.h"

TEST(SchedulerTest, memoryUpdateFunction) {
    auto const& minMaxSettings = storm::settings::getModule<storm::settings::modules::MinMaxEquationSolverSettings>();
    auto minMaxEquationSolvingTechnique = minMaxSettings.getMinMaxEquationSolvingMethod();
    if (minMaxEquationSolvingTechnique != storm::solver::MinMaxMethod::PolicyIteration) {
        storm::settings::mutableManager().setFromString("--minmax:method pi");
    }

    std::string prismModelPath = STORM_TEST_RESOURCES_DIR "/mdp/window_mp_par.nm";

    storm::prism::Program program = storm::parser::PrismParser::parse(prismModelPath);
    storm::builder::BuilderOptions options = storm::builder::BuilderOptions(true, true);
    std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(program, options).build();
    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();

    sw::FixedWindow::FixedWindowParityObjective<double> objective(*mdp, "priorities", 3);
    sw::storage::ValuesAndScheduler<double> valuesAndScheduler = sw::FixedWindow::performMaxProb(objective, true, true);
    storm::storage::Scheduler<double> scheduler = *valuesAndScheduler.scheduler;

    ASSERT_EQ(scheduler.getChoice(1, 0).getDeterministicChoice(), 0);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, 0, 1, 0, 1), 0);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, 0, 1, 0, 3), 4);
}

TEST(SchedulerTest, meanPayoffWindowGameTest) {
    std::string tra_file = STORM_TEST_RESOURCES_DIR "/tra/dfwMPGame.tra";
    std::string rew_file = STORM_TEST_RESOURCES_DIR "/rew/dfwMPGame.trans.rew";
    std::string lab_file = STORM_TEST_RESOURCES_DIR "/lab/dfwMPGame.lab";
    std::string choicelab_file = STORM_TEST_RESOURCES_DIR "/lab/dfwMPGame.choicelab";
    storm::models::sparse::Mdp<double> mdp(
            storm::parser::NondeterministicModelParser<>::parseMdp(tra_file, lab_file, "", rew_file, choicelab_file));
    storm::models::sparse::StandardRewardModel<double> rewardModel = mdp.getUniqueRewardModel();
    rewardModel.reduceToStateBasedRewards(mdp.getTransitionMatrix());
    mdp.addRewardModel("weights", storm::models::sparse::StandardRewardModel<double>(boost::none,
                                                                                     rewardModel.getStateActionRewardVector()));
    // sw::util::graphviz::GraphVizBuilder::mdpGraphExport(mdp, "", "weights");
    sw::FixedWindow::FixedWindowMeanPayoffObjective<double> objective(mdp, "weights", 3);
    storm::storage::BitVector allStates(mdp.getNumberOfStates(), true);
    sw::storage::ValuesAndScheduler<double> valuesAndScheduler = sw::FixedWindow::performMaxProb(objective, true);
    storm::storage::Scheduler<double> scheduler = *valuesAndScheduler.scheduler;
    ASSERT_EQ(scheduler.getNumberOfMemoryStates(), 3);
    uint_fast64_t currentMemory = 0;
    storm::storage::SchedulerChoice<double> choice = scheduler.getChoice(0, 0);
    if (choice.getDeterministicChoice() == 0) {
        std::vector<uint_fast64_t> currentState = {0, 1, 2, 3, 0};
        for (uint_fast64_t step = 0; step < 4; ++step) {
            choice = scheduler.getChoice(currentState[step], currentMemory);
            currentMemory = scheduler.getMemoryStructure()->getSuccessorMemoryState(mdp, currentMemory, currentState[step], choice.getDeterministicChoice(), currentState[step + 1]);
        }
        choice = scheduler.getChoice(0, currentMemory);
        ASSERT_EQ(choice.getDeterministicChoice(), 1);
        currentState = {0, 4, 5, 6, 0};
        for (uint_fast64_t step = 0; step < 4; ++step) {
            choice = scheduler.getChoice(currentState[step], currentMemory);
            currentMemory = scheduler.getMemoryStructure()->getSuccessorMemoryState(mdp, currentMemory, currentState[step], choice.getDeterministicChoice(), currentState[step + 1]);
        }
        choice = scheduler.getChoice(0, currentMemory);
        ASSERT_EQ(choice.getDeterministicChoice(), 0);
    } else {
        std::vector<uint_fast64_t> currentState = {0, 4, 5, 6, 0};
        for (uint_fast64_t step = 0; step < 4; ++step) {
            choice = scheduler.getChoice(currentState[step], currentMemory);
            currentMemory = scheduler.getMemoryStructure()->getSuccessorMemoryState(mdp, currentMemory, currentState[step], choice.getDeterministicChoice(), currentState[step + 1]);
        }
        choice = scheduler.getChoice(0, currentMemory);
        ASSERT_EQ(choice.getDeterministicChoice(), 0);
        currentState = {0, 1, 2, 3, 0};
        choice = scheduler.getChoice(0, currentMemory);
        for (uint_fast64_t step = 0; step < 4; ++step) {
            choice = scheduler.getChoice(currentState[step], currentMemory);
            currentMemory = scheduler.getMemoryStructure()->getSuccessorMemoryState(mdp, currentMemory, currentState[step], choice.getDeterministicChoice(), currentState[step + 1]);
        }
        choice = scheduler.getChoice(0, currentMemory);
        ASSERT_EQ(choice.getDeterministicChoice(), 1);
    }
}
