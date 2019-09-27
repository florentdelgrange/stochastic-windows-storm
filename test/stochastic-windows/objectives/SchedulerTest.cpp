//
// Created by florent on 23/09/19.
//

#include <storm/models/sparse/Mdp.h>
#include <storm/settings/modules/MinMaxEquationSolverSettings.h>
#include <storm-parsers/parser/PrismParser.h>
#include <stochastic-windows/directfixedwindow/DirectFixedWindowObjective.h>
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

    uint_fast64_t currentMemoryState, expectedArrivalMemoryState;
    currentMemoryState = scheduler.getMemoryStructure()->getInitialMemoryStates()[0];
    expectedArrivalMemoryState = currentMemoryState;
    ASSERT_EQ(scheduler.getChoice(1, 0).getDeterministicChoice(), 0);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, 1, 0, 1), expectedArrivalMemoryState);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(3, 0)").getNextSetIndex(0);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, 1, 0, 3), expectedArrivalMemoryState);
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

TEST(SchedulerTest, parityWindowGameTest) {
    auto const& minMaxSettings = storm::settings::getModule<storm::settings::modules::MinMaxEquationSolverSettings>();
    auto minMaxEquationSolvingTechnique = minMaxSettings.getMinMaxEquationSolvingMethod();
    if (minMaxEquationSolvingTechnique != storm::solver::MinMaxMethod::PolicyIteration) {
        storm::settings::mutableManager().setFromString("--minmax:method pi");
    }

    std::string prismModelPath = STORM_TEST_RESOURCES_DIR "/mdp/window_par_memory_example.nm";
    std::string stateRewardModel = "priorities";

    storm::prism::Program program = storm::parser::PrismParser::parse(prismModelPath);
    storm::builder::BuilderOptions options = storm::builder::BuilderOptions(true, true);
    std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(program, options).build();
    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();

    sw::FixedWindow::FixedWindowParityObjective<double> objective(*mdp, stateRewardModel, 5);
    sw::storage::ValuesAndScheduler<double> valuesAndScheduler = sw::FixedWindow::performMaxProb(objective, true, true);
    storm::storage::Scheduler<double> scheduler = *valuesAndScheduler.scheduler;
    // storm::storage::BitVector initialStates(mdp->getNumberOfStates(), false);
    // initialStates.set(0, true);
    // storm::storage::BitVector enabledActions(mdp->getNumberOfChoices(), true);
    // sw::util::graphviz::GraphVizBuilder::mdpUnfoldingExport(mdp->getTransitionMatrix(), sw::DirectFixedWindow::WindowUnfoldingParity<double>(*mdp, stateRewardModel, 5, initialStates, enabledActions));
    // sw::util::graphviz::GraphVizBuilder::mdpGraphExport(*mdp, stateRewardModel, "", "windowParityGame");
    // sw::util::graphviz::GraphVizBuilder::schedulerExport(*mdp, scheduler, "windowParityGame_schedulerProduct");

    auto getChoice = [&] (uint_fast64_t state, uint_fast64_t action) -> std::uint_fast64_t {
        return action - mdp->getTransitionMatrix().getRowGroupIndices()[state];
    };
    uint_fast64_t currentMemoryState, expectedArrivalMemoryState, state, choice, nextState;

    state = 6;
    nextState = 8;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(5, 3)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(4, 0)").getNextSetIndex(0);
    choice = getChoice(state, 6);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    nextState = 10;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(1, 1)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(1, 2)").getNextSetIndex(0);
    choice = getChoice(state, 8);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);


    nextState = 9;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(3, 2)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(3, 3)").getNextSetIndex(0);
    choice = getChoice(state, 7);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);
}

TEST(SchedulerTest, directFixedWindowMeanPayoffSchedulerTest) {
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

    sw::DirectFixedWindow::DirectFixedWindowMeanPayoffObjective<double> objective(*mdp, "weights", 3);
    sw::storage::ValuesAndScheduler<double> valuesAndScheduler = sw::DirectFixedWindow::performMaxProb(mdp->getInitialStates(), objective, true, true);
    storm::storage::Scheduler<double> scheduler = *valuesAndScheduler.scheduler;

    auto getChoice = [&] (uint_fast64_t state, uint_fast64_t action) -> std::uint_fast64_t {
        return action - mdp->getTransitionMatrix().getRowGroupIndices()[state];
    };
    uint_fast64_t currentMemoryState, expectedArrivalMemoryState, state, choice, nextState;

    state = 0;
    nextState = 2;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(0, 0)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(-5, 1)").getNextSetIndex(0);
    choice = getChoice(state, 1);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 2;
    nextState = 5;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(-5, 1)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(0, 0)").getNextSetIndex(0);
    choice = getChoice(state, 3);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);
    nextState = 4;
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 5;
    nextState = 8;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(0, 0)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(-5, 1)").getNextSetIndex(0);
    choice = getChoice(state, 7);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);
    nextState = 10;
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 8;
    nextState = 5;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(-5, 1)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(-4, 2)").getNextSetIndex(0);
    choice = getChoice(state, 12);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 10;
    nextState = 4;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(-5, 1)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(0, 0)").getNextSetIndex(0);
    choice = getChoice(state, 15);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);
    nextState = 8;
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);
    nextState = 9;
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 4;
    nextState = 7;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(0, 0)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(-5, 1)").getNextSetIndex(0);
    choice = getChoice(state, 5);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 5;
    nextState = 4;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(-4, 2)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("⊥").getNextSetIndex(0);
    choice = getChoice(state, 6);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);
    nextState = 8;
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);
    nextState = 9;
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 8;
    nextState = 5;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(0, 0)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(0, 0)").getNextSetIndex(0);
    choice = getChoice(state, 12);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 9;
    nextState = 9;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(0, 0)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(-1, 1)").getNextSetIndex(0);
    choice = getChoice(state, 13);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 9;
    nextState = 9;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(-1, 1)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(-2, 2)").getNextSetIndex(0);
    choice = getChoice(state, 13);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 9;
    nextState = 9;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(-2, 2)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("⊥").getNextSetIndex(0);
    choice = getChoice(state, 13);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);
}

TEST(SchedulerTest, directFixedWindowMeanParitySchedulerTest) {
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

    sw::DirectFixedWindow::DirectFixedWindowParityObjective<double> objective(*mdp, "priorities", 3);
    sw::storage::ValuesAndScheduler<double> valuesAndScheduler = sw::DirectFixedWindow::performMaxProb(mdp->getInitialStates(), objective, true, true);
    storm::storage::Scheduler<double> scheduler = *valuesAndScheduler.scheduler;
    // sw::util::graphviz::GraphVizBuilder::schedulerExport(*mdp, scheduler, "direct_fixed_par");

    auto getChoice = [&] (uint_fast64_t state, uint_fast64_t action) -> std::uint_fast64_t {
        return action - mdp->getTransitionMatrix().getRowGroupIndices()[state];
    };
    uint_fast64_t currentMemoryState, expectedArrivalMemoryState, state, choice, nextState;

    state = 0;
    nextState = 2;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(3, 0)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(0, 0)").getNextSetIndex(0);
    choice = getChoice(state, 1);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 2;
    nextState = 5;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(0, 0)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(3, 0)").getNextSetIndex(0);
    choice = getChoice(state, 3);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);
    nextState = 4;
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(0, 0)").getNextSetIndex(0);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 5;
    nextState = 8;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(3, 0)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(1, 1)").getNextSetIndex(0);
    choice = getChoice(state, 6);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);
    nextState = 9;
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);
    nextState = 4;
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(0, 0)").getNextSetIndex(0);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 8;
    nextState = 5;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(1, 1)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("⊥").getNextSetIndex(0);
    choice = getChoice(state, 12);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 9;
    nextState = 9;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(1, 1)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("⊥").getNextSetIndex(0);
    choice = getChoice(state, 13);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 4;
    nextState = 7;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(0, 0)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(3, 0)").getNextSetIndex(0);
    choice = getChoice(state, 5);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);

    state = 7;
    nextState = 4;
    currentMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(3, 0)").getNextSetIndex(0);
    expectedArrivalMemoryState = scheduler.getMemoryStructure()->getStateLabeling().getStates("(0, 0)").getNextSetIndex(0);
    choice = getChoice(state, 10);
    ASSERT_EQ(scheduler.getChoice(state, currentMemoryState).getDeterministicChoice(), choice);
    ASSERT_EQ(scheduler.getMemoryStructure()->getSuccessorMemoryState(*mdp, currentMemoryState, state, choice, nextState), expectedArrivalMemoryState);
}
