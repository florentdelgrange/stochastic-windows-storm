//
// Created by Florent Delgrange on 25/10/2018.
//
#include <time.h>

#include "storm-config.h"
#include "storm-parsers/parser/AutoParser.h"
#include "storm/storage/MaximalEndComponentDecomposition.h"
#include "storm/models/sparse/MarkovAutomaton.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/models/sparse/StandardRewardModel.h"
#include "storm/builder/ExplicitModelBuilder.h"
#include "storm/storage/SymbolicModelDescription.h"
#include "storm-parsers/parser/PrismParser.h"
#include <iostream>
#include <stochastic-windows/prefixindependent/MaximalEndComponentDecompositionUnfolding.h>
#include <stochastic-windows/directfixedwindow/DirectFixedWindowObjective.h>
#include <stochastic-windows/game/WindowGame.h>
#include <stochastic-windows/game/PredecessorsSquaredLinkedList.h>
#include <stochastic-windows/prefixindependent/MaximalEndComponentDecompositionWindowGame.h>
#include <stochastic-windows/fixedwindow/MaximalEndComponentClassifier.h>
#include <stochastic-windows/fixedwindow/FixedWindowObjective.h>
#include <stochastic-windows/game/TotalPayoffGame.h>
#include <stochastic-windows/game/WeakParityGame.h>
#include <stochastic-windows/boundedwindow/BoundedWindowObjective.h>

#include "storm/utility/initialize.h"

#include "storm/settings/modules/GeneralSettings.h"
#include "storm/settings/modules/CoreSettings.h"
#include "storm/settings/modules/IOSettings.h"
#include "storm/settings/modules/DebugSettings.h"
#include "storm/settings/modules/CuddSettings.h"
#include "storm/settings/modules/SylvanSettings.h"
#include "storm/settings/modules/EigenEquationSolverSettings.h"
#include "storm/settings/modules/GmmxxEquationSolverSettings.h"
#include "storm/settings/modules/NativeEquationSolverSettings.h"
#include "storm/settings/modules/EliminationSettings.h"
#include "storm/settings/modules/MinMaxEquationSolverSettings.h"
#include "storm/settings/modules/GameSolverSettings.h"
#include "storm/settings/modules/BisimulationSettings.h"
#include "storm/settings/modules/GlpkSettings.h"
#include "storm/settings/modules/GurobiSettings.h"
#include "storm/settings/modules/Smt2SmtSolverSettings.h"
#include "storm/settings/modules/ExplorationSettings.h"
#include "storm/settings/modules/ResourceSettings.h"
#include "storm/settings/modules/AbstractionSettings.h"
#include "storm/settings/modules/BuildSettings.h"
#include "storm/settings/modules/JitBuilderSettings.h"
#include "storm/settings/modules/MultiObjectiveSettings.h"
#include "storm/settings/modules/TopologicalEquationSolverSettings.h"
#include "storm/settings/modules/MultiplierSettings.h"

#include <storm/environment/solver/MinMaxSolverEnvironment.h>

#include "storm/analysis/GraphConditions.h"
#include <storm/utility/builder.h>
#include <storm/storage/sparse/ModelComponents.h>
#include <storm/models/sparse/StateLabeling.h>

#include "storm-cli-utilities/cli.h"
#include "storm-cli-utilities/model-handling.h"

#include "storm/api/storm.h"

#include "stochastic-windows/util/Graphviz.h"


std::string minMaxMethodAsString() {
    auto const& minMaxSettings = storm::settings::getModule<storm::settings::modules::MinMaxEquationSolverSettings>();
    auto minMaxEquationSolvingTechnique = minMaxSettings.getMinMaxEquationSolvingMethod();
    switch(minMaxEquationSolvingTechnique){
        case storm::solver::MinMaxMethod::ValueIteration: return "Value Iteration";
        case storm::solver::MinMaxMethod::PolicyIteration:  return "Policy Iteration";
        case storm::solver::MinMaxMethod::LinearProgramming: return "Linear Programming";
        case storm::solver::MinMaxMethod::RationalSearch: return "Rational Serarch";
        case storm::solver::MinMaxMethod::IntervalIteration: return "Interval Iteration";
        case storm::solver::MinMaxMethod::SoundValueIteration: return "Sound Value Iteration";
        case storm::solver::MinMaxMethod::Topological: return "Topological";
        default: return "";
    }
}

/*!
 * Initialize the settings manager.
 */
void initializeSettings() {
    storm::settings::mutableManager().setName("Stochastic Windows", "stochastic-windows");

    storm::settings::addModule<storm::settings::modules::GeneralSettings>();
    storm::settings::addModule<storm::settings::modules::IOSettings>();
    storm::settings::addModule<storm::settings::modules::CoreSettings>();
    storm::settings::addModule<storm::settings::modules::DebugSettings>();
    storm::settings::addModule<storm::settings::modules::BuildSettings>();
    storm::settings::addModule<storm::settings::modules::NativeEquationSolverSettings>();
    storm::settings::addModule<storm::settings::modules::EliminationSettings>();
    storm::settings::addModule<storm::settings::modules::MinMaxEquationSolverSettings>();
    storm::settings::addModule<storm::settings::modules::GlpkSettings>();
    storm::settings::addModule<storm::settings::modules::ExplorationSettings>();
    storm::settings::addModule<storm::settings::modules::TopologicalEquationSolverSettings>();
    storm::settings::addModule<storm::settings::modules::ResourceSettings>();
    storm::settings::addModule<storm::settings::modules::GmmxxEquationSolverSettings>();
    storm::settings::addModule<storm::settings::modules::MultiplierSettings>();

    // DEBUG MODE
    storm::utility::setLogLevel(l3pp::LogLevel::DEBUG);

    storm::settings::mutableManager().printHelpForModule("minmax");
    storm::settings::mutableManager().setFromString("--minmax:method pi");
    std::cout << "Equation solving method: " << minMaxMethodAsString() << std::endl;
}


void mecDecompositionPrintExamples() {

        std::string prismModelPath = STORM_TEST_RESOURCES_DIR "/mdp/sensors.prism";
        storm::storage::SymbolicModelDescription modelDescription = storm::parser::PrismParser::parse(prismModelPath);
        storm::prism::Program program = modelDescription.preprocess().asPrismProgram();
        storm::builder::BuilderOptions options = storm::builder::BuilderOptions(true, true);
        std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(program, options).build();

        std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();
        mdp->printModelInformationToStream(std::cout);
        storm::models::sparse::StandardRewardModel<double> rewardModel = model->getRewardModel("energy");

        storm::storage::MaximalEndComponentDecomposition<double> mecDecomposition(*mdp);

        std::cout << "MDP 1: Choice Matrix" << std::endl;
        std::cout << mdp->getTransitionMatrix() << std::endl;
        std::cout << "Number of states: ";
        std::cout << mdp->getNumberOfStates() << std::endl;
        std::cout << "Reward model ? ";
        std::cout << program.getNumberOfRewardModels() << " ";
        std::cout << "Number of transitions: ";
        std::cout << mdp->getNumberOfTransitions() << std::endl;
        std::cout << "Number of choices: ";
        std::cout << mdp->getNumberOfChoices() << std::endl;

        std::cout << "MEC decomposition: ";
        std::cout << mecDecomposition << std::endl;

        std::cout << "Reward Model" << std::endl;
        std::cout << rewardModel << std::endl;
        std::vector<double> rewardVector = rewardModel.getStateActionRewardVector();
        // Note that each row of the transition matrix corresponds to an action in the state associated with the linked
        // row group.
        uint_fast64_t row = 0;
        for (auto reward : rewardVector) {
            std::cout << "action " << row << ": reward=" << reward << std::endl;
            row++;
        }
        std::cout << std::endl;

        // create a vector of size |MECs|
        // std::vector<storm::storage::SparseMatrix<double>> matrices(mecDecomposition.size());

        storm::storage::SparseMatrix<double> originalMatrix = mdp->getTransitionMatrix();
        for(std::vector<storm::storage::MaximalEndComponent>::const_iterator mec = mecDecomposition.begin();
                mec != mecDecomposition.end(); ++mec){
            std::cout << "Mec -> " << *mec << std::endl;
            storm::storage::MaximalEndComponent::set_type stateSet = mec->getStateSet();
            for (storm::storage::MaximalEndComponent::set_type::const_iterator state = stateSet.begin();
                    state != stateSet.end(); ++state){
                std::cout << "state " << *state << "-> actions = ";
                storm::storage::MaximalEndComponent::set_type actionSet = mec->getChoicesForState(*state);
                for (storm::storage::MaximalEndComponent::set_type::const_iterator action = actionSet.begin();
                         action != actionSet.end(); ++action) {
                     std::cout << *action << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
        std::cout << "SPARSE MATRIX" << std::endl;
        std::cout << " by iterating on rows with getRowGroupIndices() " << std::endl;
        std::vector<storm::storage::SparseMatrix< double >::index_type> groups = originalMatrix.getRowGroupIndices();
        for (uint_fast64_t s = 0; s < mdp->getNumberOfStates(); ++s){
            std::cout << "State " << s << std::endl;
            for (uint_fast64_t row = groups[s]; row < groups[s + 1]; ++ row) {
                std::cout << "  row number " << row << " ---> ";
                for (auto entry : originalMatrix.getRow(row)) {
                    std::cout <<  "p=" << entry.getValue() << ": s'=" << entry.getColumn() << "; ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << " -------------------- " << std::endl;
        std::cout << " with getRowGroup(state) " << std::endl;
        for (uint_fast64_t s = 0; s < mdp->getNumberOfStates(); ++s){
            std::cout << "State " << s << std::endl;
            std::cout << "    ";
            for (auto entry : originalMatrix.getRowGroup(s)) {
                    std::cout << entry << ", ";
                }
                std::cout << std::endl;
        }

    std::cout << std::endl;
    std::cout << std::endl;
    // now some examples about vectors
    std::cout << "Vectors initialization example" << std::endl;
    std::vector<std::vector<std::vector<std::tuple<uint_fast64_t, double>>>> newRowGroupEntries;
    std::cout << "init : std::vector<std::vector<std::vector<std::tuple<uint_fast64_t, double>>>> newRowGroupEntries;" << std::endl;
    std::cout << "declaration, newRowGroupEntries.size()=" << newRowGroupEntries.size() << std::endl;
    newRowGroupEntries.emplace_back();
    std::cout << "newRowGroupEntries.emplace_back();" << std::endl;
    std::cout << "=> add an empty vector in it; newRowGroupEntries.size()=" << newRowGroupEntries.size() << std::endl;
    std::cout << "newRowGroupEntries[0].size()=" << newRowGroupEntries[0].size() << std::endl;

};

void schedulersExamples(){
    sw::storage::SchedulerProductLabeling schedulerLabelingWeights;
    sw::storage::SchedulerProductLabeling schedulerLabelingPriorities;

    schedulerLabelingWeights.weights = "weights";
    schedulerLabelingPriorities.priorities = "priorities";

    {
        std::string prismModelPath = STORM_SOURCE_DIR "/src/stochastic-windows/util/graphviz-examples/dfwMemory.prism";
        storm::storage::SymbolicModelDescription modelDescription = storm::parser::PrismParser::parse(prismModelPath);
        storm::prism::Program program = modelDescription.preprocess().asPrismProgram();
        storm::builder::BuilderOptions options = storm::builder::BuilderOptions(true, true);
        std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(
                program, options).build();

        std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();
        mdp->printModelInformationToStream(std::cout);
        std::cout << mdp->getTransitionMatrix() << std::endl;

        storm::storage::BitVector restrictedStateSpace(mdp->getNumberOfStates(), true);
        storm::storage::BitVector enabledActions(mdp->getNumberOfChoices(), true);
        std::unique_ptr<sw::game::WindowGame<double>>
                wmpGame = std::unique_ptr<sw::game::WindowGame<double>>(
                new sw::game::WindowMeanPayoffGame<double>(*mdp, "weights", 3, restrictedStateSpace, enabledActions)
        );
        std::cout << "DFW scheduler: memory requirements" << std::endl;
        sw::game::WinningSetAndScheduler<double> winningSetAndScheduler = wmpGame->produceSchedulerForDirectFW(true);
        //std::cout << winningSetAndScheduler.winningSet << std::endl;
        std::cout << winningSetAndScheduler.scheduler->getMemoryStructure()->toString() << std::endl;
        winningSetAndScheduler.scheduler->printToStream(std::cout, mdp);

        // Graphviz
        storm::storage::SparseMatrix<double> matrix = mdp->getTransitionMatrix();
        storm::models::sparse::StandardRewardModel<double> weights = model->getRewardModel("weights");
        std::vector<double> weightVector = weights.getStateActionRewardVector();

        sw::util::graphviz::GraphVizBuilder::mdpGraphWeightsExport(matrix, weightVector, "dfwMemoryExample");
        sw::util::graphviz::GraphVizBuilder::schedulerExport(*mdp, *winningSetAndScheduler.scheduler, "dfwMemoryExampleScheduler", schedulerLabelingWeights);
    }

    // schedulers
    {
        std::string prismModelPath = STORM_SOURCE_DIR "/src/stochastic-windows/util/graphviz-examples/window_mp_par.prism";
        //std::string prismModelPath = STORM_TEST_RESOURCES_DIR "/mdp/sw_simple_example.prism";
        storm::prism::Program program = storm::parser::PrismParser::parse(prismModelPath);
        storm::builder::BuilderOptions options = storm::builder::BuilderOptions(true, true);
        std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(
                program, options).build();
        std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();
        storm::storage::BitVector phiStates(mdp->getNumberOfStates(), false);
        phiStates.set(0, true); phiStates.set(5, true);
        sw::DirectFixedWindow::DirectFixedWindowParityObjective<double> dfwParObjective(*mdp, "priorities", 3);
        sw::DirectFixedWindow::DirectFixedWindowMeanPayoffObjective<double> dfwMpObjective(*mdp, "weights", 3);
        std::unique_ptr<sw::DirectFixedWindow::WindowUnfolding<double>> unfoldingDirectFixedMP = dfwMpObjective.performUnfolding(
                phiStates);
        std::unique_ptr<sw::DirectFixedWindow::WindowUnfolding<double>> unfoldingDirectFixedPar = dfwParObjective.performUnfolding(
                phiStates);
        std::cout << "DFWmp: memory structure? " << unfoldingDirectFixedMP->generateMemory().memoryStructure->toString()
                  << std::endl;
        std::cout << "DFWpar: memory structure? "
                  << unfoldingDirectFixedPar->generateMemory().memoryStructure->toString() << std::endl;
        std::cout << "DFW scheduler: memory requirements (game version)" << std::endl;
        storm::storage::BitVector restrictedStateSpace(mdp->getNumberOfStates(), true);
        storm::storage::BitVector enabledActions(mdp->getNumberOfChoices(), true);
        std::unique_ptr<sw::game::WindowGame<double>>
                wmpGame = std::unique_ptr<sw::game::WindowGame<double>>(new sw::game::WindowMeanPayoffGame<double>(*mdp, "weights", 3, restrictedStateSpace, enabledActions));
        sw::game::WinningSetAndScheduler<double> winningSetAndScheduler = wmpGame->produceSchedulerForDirectFW();
        std::cout << winningSetAndScheduler.scheduler->getMemoryStructure()->toString() << std::endl;
        winningSetAndScheduler.scheduler->printToStream(std::cout, mdp);
        std::cout << std::endl;
        std::cout << "DFW mp scheduler" << std::endl;
        sw::storage::ValuesAndScheduler<double> resultDFWmp = sw::DirectFixedWindow::performMaxProb<double>(phiStates, dfwMpObjective, true, true);
        std::cout << "[  ";
        for (double const& value : resultDFWmp.values) {
            std::cout << value << "  ";
        }
        std::cout << "]" << std::endl;
        resultDFWmp.scheduler->printToStream(std::cout, mdp);
        sw::util::graphviz::GraphVizBuilder::schedulerExport(*mdp, *resultDFWmp.scheduler, "direct_fixed_mp_unfolding", schedulerLabelingWeights);
        std::cout << std::endl;
        std::cout << "FW schedulers" << std::endl;
        sw::storage::MaximalEndComponentDecompositionWindowMeanPayoffGame<double> mecGameMP(*mdp, "weights", 3);
        sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<double> mecUnfoldingMP(*mdp, "weights", 3);
        sw::storage::MaximalEndComponentDecompositionUnfoldingParity<double> mecUnfoldingPar(*mdp, "priorities", 3);
        {
            std::cout << "Mean payoff game based classification" << std::endl;
            sw::FixedWindow::MaximalEndComponentClassifier<double> classifier(*mdp, mecGameMP, true);
            sw::FixedWindow::FixedWindowMeanPayoffObjective<double> fixedWindowObjective(*mdp, "weights", 3);
            sw::storage::ValuesAndScheduler<double> result = sw::FixedWindow::performMaxProb(fixedWindowObjective, true);
            std::cout << "[  ";
            for (double const& value : result.values) {
                std::cout << value << "  ";
            }
            std::cout << "]" << std::endl;
            result.scheduler->printToStream(std::cout, mdp);
            sw::util::graphviz::GraphVizBuilder::schedulerExport(*mdp, *result.scheduler, "fixed_mp_game", schedulerLabelingWeights);
        }
        {
            std::cout << "Mean payoff unfolding based classification" << std::endl;
            sw::FixedWindow::MaximalEndComponentClassifier<double> classifier(*mdp, mecUnfoldingMP, true);
            std::cout << classifier.getMaximalEndComponentScheduler().getMemoryStructure()->toString() << std::endl;
            sw::FixedWindow::FixedWindowMeanPayoffObjective<double> fixedWindowObjective(*mdp, "weights", 3, false);
            sw::storage::ValuesAndScheduler<double> result = sw::FixedWindow::performMaxProb(fixedWindowObjective, true, true);
            std::cout << "[  ";
            for (double const& value : result.values) {
                std::cout << value << "  ";
            }
            std::cout << "]" << std::endl;
            result.scheduler->printToStream(std::cout, mdp);
            sw::util::graphviz::GraphVizBuilder::schedulerExport(*mdp, *result.scheduler, "fixed_mp_unfolding", schedulerLabelingWeights);
        }
        {
            std::cout << "Parity unfolding based classification" << std::endl;
            sw::FixedWindow::MaximalEndComponentClassifier<double> classifier(*mdp, mecUnfoldingPar, true);
            std::cout << classifier.getMaximalEndComponentScheduler().getMemoryStructure()->toString() << std::endl;
            sw::FixedWindow::FixedWindowParityObjective<double> fixedWindowObjective(*mdp, "priorities", 3);
            sw::storage::ValuesAndScheduler<double> result = sw::FixedWindow::performMaxProb(fixedWindowObjective, true, true);
            std::cout << "[  ";
            for (double const& value : result.values) {
                std::cout << value << "  ";
            }
            std::cout << "]" << std::endl;
            result.scheduler->printToStream(std::cout, mdp);
            sw::util::graphviz::GraphVizBuilder::schedulerExport(*mdp, *result.scheduler, "fixed_par_unfolding", schedulerLabelingPriorities);
        }
        std::cout << "BW schedulers" << std::endl;
        sw::storage::MaximalEndComponentDecompositionWindowMeanPayoffGame<double> bwMPGames(*mdp, "weights");
        sw::storage::MaximalEndComponentDecompositionWindowParityGame<double> bwParGames(*mdp, "priorities");
        {
            std::cout << "Bounded Window Mean Payoff game based classification (memoryless)" << std::endl;
            clock_t start = clock();
            sw::BoundedWindow::BoundedWindowMeanPayoffObjective<double> boundedWindowObjective(*mdp, "weights");
            sw::storage::ValuesAndScheduler<double> result = sw::BoundedWindow::performMaxProb(boundedWindowObjective, true, true);

            clock_t stop = clock();
            double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
            printf("(time: %.5f)\n", elapsed);

            std::cout << "[  ";
            for (double const& value : result.values) {
                std::cout << value << "  ";
            }
            std::cout << "]" << std::endl;
            result.scheduler->printToStream(std::cout, mdp);
            sw::util::graphviz::GraphVizBuilder::schedulerExport(*mdp, *result.scheduler, "bounded_mp_memoryless", schedulerLabelingWeights);
        }
        {
            clock_t start = clock();
            std::cout << "Bounded Window Parity game based classification (memoryless)" << std::endl;
            sw::BoundedWindow::MaximalEndComponentClassifier<double> classifier(*mdp, bwParGames, true);
            sw::BoundedWindow::BoundedWindowParityObjective<double> boundedWindowObjective(*mdp, "priorities");
            sw::storage::ValuesAndScheduler<double> result = sw::BoundedWindow::performMaxProb(boundedWindowObjective, true, true);

            clock_t stop = clock();
            double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
            printf("(time: %.5f)\n", elapsed);

            std::cout << "[  ";
            for (double const& value : result.values) {
                std::cout << value << "  ";
            }
            std::cout << "]" << std::endl;
            result.scheduler->printToStream(std::cout, mdp);
            sw::util::graphviz::GraphVizBuilder::schedulerExport(*mdp, *result.scheduler, "bounded_par_memoryless", schedulerLabelingPriorities);
        }
        {
            clock_t start = clock();
            std::cout << "Bounded Window Mean Payoff game based classification (with uniform bound)" << std::endl;
            sw::BoundedWindow::MaximalEndComponentClassifier<double> classifier(*mdp, bwMPGames, true);
            sw::BoundedWindow::BoundedWindowMeanPayoffObjective<double>
            boundedWindowObjective(*mdp, "weights", sw::BoundedWindow::ClassificationMethod::WindowGameWithBound);
            sw::storage::ValuesAndScheduler<double> result = sw::BoundedWindow::performMaxProb(boundedWindowObjective, true, true);

            clock_t stop = clock();
            double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
            printf("(time: %.5f)\n", elapsed);

            std::cout << "[  ";
            for (double const& value : result.values) {
                std::cout << value << "  ";
            }
            std::cout << "]" << std::endl;
            // resulted file too large
            // sw::util::graphviz::GraphVizBuilder::schedulerExport(*mdp, *result.scheduler, "bounded_mp_game");
        }
        {
            clock_t start = clock();
            std::cout << "Bounded Window Parity game based classification (with uniform bound) = unfolding based classification" << std::endl;
            sw::BoundedWindow::BoundedWindowParityObjective<double>
            boundedWindowObjective(*mdp, "priorities", sw::BoundedWindow::ClassificationMethod::WindowGameWithBound);
            sw::storage::ValuesAndScheduler<double> result = sw::BoundedWindow::performMaxProb(boundedWindowObjective, true);

            clock_t stop = clock();
            double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
            printf("(time: %.5f)\n", elapsed);

            std::cout << "[  ";
            for (double const& value : result.values) {
                std::cout << value << "  ";
            }
            std::cout << "]" << std::endl;
            sw::util::graphviz::GraphVizBuilder::schedulerExport(*mdp, *result.scheduler, "bounded_par_game", schedulerLabelingPriorities);
        }
        /*
        {
            clock_t start = clock();
            std::cout << "Bounded Window Mean Payoff unfolding based classification" << std::endl;
            sw::BoundedWindow::MaximalEndComponentClassifier<double> classifier(*mdp, bwMPGames, true);
            sw::BoundedWindow::BoundedWindowMeanPayoffObjective<double>
            boundedWindowObjective(*mdp, "weights", sw::BoundedWindow::ClassificationMethod::Unfolding);
            sw::storage::ValuesAndScheduler<double> result = sw::BoundedWindow::performMaxProb(boundedWindowObjective);

            clock_t stop = clock();
            double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
            printf("(time: %.5f)\n", elapsed);

            std::cout << "[  ";
            for (double const& value : result.values) {
                std::cout << value << "  ";
            }
            std::cout << "]" << std::endl;
        }
        */
    }

}

void windowExamples(){
    std::string prismModelPath = STORM_SOURCE_DIR "/src/stochastic-windows/util/graphviz-examples/window_mp_par.prism";
    //std::string prismModelPath = STORM_TEST_RESOURCES_DIR "/mdp/sw_simple_example.prism";
    storm::prism::Program program = storm::parser::PrismParser::parse(prismModelPath);
    storm::builder::BuilderOptions options = storm::builder::BuilderOptions(true, true);
    std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(program, options).build();
    std::cout << model->hasChoiceOrigins() << std::endl;

    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();
    storm::storage::MaximalEndComponentDecomposition<double> mecDecomposition(*mdp);

    mdp->printModelInformationToStream(std::cout);
    std::cout << mdp->getTransitionMatrix() << std::endl;
    // std::cout << mdp->getTransitionMatrix().transpose() << std::endl;
    // std::cout << mdp->getTransitionMatrix().transpose(true) << std::endl;

    std::cout << mecDecomposition << std::endl;

    // maximum window size is 3
    // Mean Payoff
    sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<double> unfoldingMp(*mdp, "weights", 3);
    std::cout << "unfolded matrices Mean Payoff: " << endl;
    // Parity
    sw::storage::MaximalEndComponentDecompositionUnfoldingParity<double> unfoldingPar(*mdp, "priorities", 3);
    std::cout << "unfolded matrices Parity: " << endl;

    // DirectFixed MP
    sw::DirectFixedWindow::DirectFixedWindowMeanPayoffObjective<double> dfwMpObjective(*mdp, "weights", 3);
    storm::storage::BitVector phiStates(mdp->getNumberOfStates(), false);
    phiStates.set(0, true); phiStates.set(5, true);
    std::unique_ptr<sw::DirectFixedWindow::WindowUnfolding<double>> unfoldingDirectFixedMP = dfwMpObjective.performUnfolding(phiStates);
    sw::storage::ValuesAndScheduler<double> resultDFWmp = sw::DirectFixedWindow::performMaxProb<double>(phiStates, dfwMpObjective, true);
    std::cout << "Pr(DFWmp) = [";
    for (auto state: phiStates) {
        std::cout << "s" << state << "=" << resultDFWmp.values[state] << ", ";
    }
    std::cout << "]" << std::endl;
    // DirectFixed Par
    sw::DirectFixedWindow::DirectFixedWindowParityObjective<double> dfwParObjective(*mdp, "priorities", 3);
    phiStates = storm::storage::BitVector(mdp->getNumberOfStates(), true);
    std::unique_ptr<sw::DirectFixedWindow::WindowUnfolding<double>> unfoldingDirectFixedPar = dfwParObjective.performUnfolding(phiStates);
    sw::storage::ValuesAndScheduler<double> result = sw::DirectFixedWindow::performMaxProb<double>(phiStates, dfwParObjective);
    std::cout << "Pr(DFWpar) = [";
    for (auto state: phiStates) {
        std::cout << "s" << state << "=" << result.values[state] << ", ";
    }
    std::cout << "]" << std::endl;
    phiStates.clear(); phiStates.set(5);
    std::cout << "Pr_s5(DFWpar) = " << sw::DirectFixedWindow::performMaxProb<double>(phiStates, dfwParObjective).values[5] << std::endl;
    std::cout << std::endl;

    // Window Mean Payoff game
    std::cout << "window mean payoff game" << std::endl;
    storm::storage::BitVector restrictedStateSpace(mdp->getNumberOfStates(), true);
    storm::storage::BitVector enabledActions(mdp->getNumberOfChoices(), true);
    std::unique_ptr<sw::game::WindowGame<double>>
    wmpGame = std::unique_ptr<sw::game::WindowGame<double>>(new sw::game::WindowMeanPayoffGame<double>(*mdp, "weights", 3, restrictedStateSpace, enabledActions));
    std::cout << "Direct Fixed Window winning set in the whole MDP: " << wmpGame->directFW() << std::endl;
    std::cout << std::endl;

    // Graphviz
    storm::storage::SparseMatrix<double> matrix = mdp->getTransitionMatrix();
    storm::models::sparse::StandardRewardModel<double> weights = model->getRewardModel("weights");
    storm::models::sparse::StandardRewardModel<double> priorities = model->getRewardModel("priorities");
    std::vector<double> weightVector = weights.getStateActionRewardVector();
    std::vector<double> priorityVector = priorities.getStateRewardVector();

    sw::util::graphviz::GraphVizBuilder::mdpGraphExport(matrix, weightVector, priorityVector);
    sw::util::graphviz::GraphVizBuilder::unfoldedECsExport(matrix, unfoldingMp, "mp");
    sw::util::graphviz::GraphVizBuilder::unfoldedECsExport(matrix, unfoldingPar, "par");
    sw::util::graphviz::GraphVizBuilder::mdpUnfoldingExport(matrix, *unfoldingDirectFixedMP, "direct_fixed_mp");
    sw::util::graphviz::GraphVizBuilder::mdpUnfoldingExport(matrix, *unfoldingDirectFixedPar, "direct_fixed_par");

    // MEC classification
    std::cout << "MEC classification" << std::endl;
    std::cout << "Classification by unfolding MECs (MP)" << std::endl;
    sw::FixedWindow::MaximalEndComponentClassifier<double> classifierUnfoldingMP(*mdp, unfoldingMp);
    std::cout << "Safe states " << classifierUnfoldingMP.getSafeStateSpace() << std::endl;
    std::cout << "Good states " << classifierUnfoldingMP.getGoodStateSpace() << std::endl;
    std::cout << "Classification by unfolding MECs (Par)" << std::endl;
    sw::FixedWindow::MaximalEndComponentClassifier<double> classifierUnfoldingPar(*mdp, unfoldingPar);
    std::cout << "Safe states " << classifierUnfoldingPar.getSafeStateSpace() << std::endl;
    std::cout << "Good states " << classifierUnfoldingPar.getGoodStateSpace() << std::endl;
    std::cout << "Classification by considering MECs as games (MP)" << std::endl;
    sw::storage::MaximalEndComponentDecompositionWindowMeanPayoffGame<double> games(*mdp, "weights", 3);
    sw::FixedWindow::MaximalEndComponentClassifier<double> gameClassifier(*mdp, games);
    std::cout << "Safe states " << gameClassifier.getSafeStateSpace() << std::endl;
    std::cout << "Good states " << gameClassifier.getGoodStateSpace() << std::endl;
    std::cout << std::endl;

    // Fixed Window Objective
    std::cout << "Fixed Window Objectives: mean payoff (with game classification)" << std::endl;
    sw::FixedWindow::FixedWindowMeanPayoffObjective<double> fixedWindowMPObjectiveGame(*mdp, "weights", 3);
    sw::storage::ValuesAndScheduler<double> fwResult = sw::FixedWindow::performMaxProb(fixedWindowMPObjectiveGame);
    std::cout << "Pr(FWmp) = [";
    for (uint_fast64_t state = 0; state < mdp->getNumberOfStates(); ++ state) {
        std::cout << "s" << state << "=" << fwResult.values[state] << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Fixed Window Objectives: mean payoff (with unfolding based classification)" << std::endl;
    sw::FixedWindow::FixedWindowMeanPayoffObjective<double> fixedWindowMPObjectiveUnfolding(*mdp, "weights", 3, false);
     fwResult = sw::FixedWindow::performMaxProb(fixedWindowMPObjectiveUnfolding);
    std::cout << "Pr(FWmp) = [";
    for (uint_fast64_t state = 0; state < mdp->getNumberOfStates(); ++ state) {
        std::cout << "s" << state << "=" << fwResult.values[state] << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Fixed Window Objectives: parity" << std::endl;
    sw::FixedWindow::FixedWindowParityObjective<double> fixedWindowParityObjective(*mdp, "priorities", 3);
    fwResult = sw::FixedWindow::performMaxProb(fixedWindowParityObjective);
    std::cout << "Pr(FWpar) = [";
    for (uint_fast64_t state = 0; state < mdp->getNumberOfStates(); ++ state) {
        std::cout << "s" << state << "=" << fwResult.values[state] << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;

    // Total Payoff
    std::cout << std::endl;
    std::cout << "MDP as a Total Payoff game" << std::endl;
    sw::game::TotalPayoffGame<double> game(*mdp, "weights", restrictedStateSpace, enabledActions);
    /* */
    {
        std::vector<double> values = game.maxTotalPayoffInf();
        std::cout << "max total payoff inf values= [";
        for (uint_fast64_t state: restrictedStateSpace) {
            std::cout << "s_" << state << "=" << values[state] << ", ";
        }
        std::cout << "]" << std::endl;
        values = game.minTotalPayoffSup();
        std::cout << "min total payoff sup values= [";
        for (uint_fast64_t state: restrictedStateSpace) {
            std::cout << "s_" << state << "=" << values[state] << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << std::endl;
    }
     /* */

    // Attractors
    // P1
    for (uint_fast64_t state: restrictedStateSpace) {
        storm::storage::BitVector T(restrictedStateSpace.size(), false);
        T.set(state, true);
        sw::game::BackwardTransitions backwardTransitions;
        game.initBackwardTransitions(backwardTransitions);
        std::cout << "P1 attractors of {" << state << "}: ";
        std::cout << game.attractorsP1(T, backwardTransitions) << std::endl;
    }
    {
        storm::storage::BitVector T(restrictedStateSpace.size(), false);
        T.set(4, true); T.set(7, true); T.set(11, true);
        sw::game::BackwardTransitions backwardTransitions;
        game.initBackwardTransitions(backwardTransitions);
        std::cout << "P1 attractors of T = {4, 7, 11}: ";
        std::cout << game.attractorsP1(T, backwardTransitions) << std::endl;
    }
    {
        storm::storage::BitVector T(restrictedStateSpace.size(), false);
        T.set(5, true); T.set(4, true);
        sw::game::BackwardTransitions backwardTransitions;
        game.initBackwardTransitions(backwardTransitions);
        std::cout << "P1 attractors of T = {5, 4}: ";
        std::cout << game.attractorsP1(T, backwardTransitions) << std::endl;
    }

    // Attractors
    // P1
    for (uint_fast64_t state: restrictedStateSpace) {
        storm::storage::BitVector T(restrictedStateSpace.size(), false);
        T.set(state, true);
        sw::game::BackwardTransitions backwardTransitions;
        game.initBackwardTransitions(backwardTransitions);
        std::cout << "P1 attractors of {" << state << "}: ";
        std::cout << game.attractorsP1(T, backwardTransitions) << std::endl;
    }
    {
        storm::storage::BitVector T(restrictedStateSpace.size(), false);
        T.set(4, true); T.set(7, true); T.set(11, true);
        sw::game::BackwardTransitions backwardTransitions;
        game.initBackwardTransitions(backwardTransitions);
        std::cout << "P1 attractors of T = {4, 7, 11}: ";
        std::cout << game.attractorsP1(T, backwardTransitions) << std::endl;
    }
    {
        storm::storage::BitVector T(restrictedStateSpace.size(), false);
        T.set(5, true); T.set(4, true);
        sw::game::BackwardTransitions backwardTransitions;
        game.initBackwardTransitions(backwardTransitions);
        std::cout << "P1 attractors of T = {5, 4}: ";
        std::cout << game.attractorsP1(T, backwardTransitions) << std::endl;
    }
    {
        sw::game::GameStates S;
        S.p1States = storm::storage::BitVector(restrictedStateSpace.size(), false);
        S.p2States = storm::storage::BitVector(enabledActions.size(), false);
        S.p1States.set(7, true); S.p2States.set(9, true);
        sw::game::BackwardTransitions backwardTransitions;
        game.initBackwardTransitions(backwardTransitions);
        std::cout << "P1 attractors of  {s7, a9}: ";
        sw::game::GameStates T = game.attractorsP1(S, backwardTransitions);
        std::cout << "S1=" << T.p1States << ", S2=" << T.p2States << std::endl;
    }
    // P2
    std::cout << std::endl;
    for (uint_fast64_t state: restrictedStateSpace) {
        sw::game::GameStates S;
        S.p1States = storm::storage::BitVector(restrictedStateSpace.size(), false);
        S.p2States = storm::storage::BitVector(enabledActions.size(), false);
        S.p1States.set(state, true);
        std::cout << "P2 attractors of {" << state << "}" << std::endl;
        sw::game::BackwardTransitions backwardTransitions;
        game.initBackwardTransitions(backwardTransitions);
        sw::game::GameStates T = game.attractorsP2(S, backwardTransitions);
        std::cout << "S1=" << T.p1States << ", S2=" << T.p2States << std::endl;
    }

    {
        std::cout << R"(With "non-trivial" restricted state space S\{9})" << std::endl;
        sw::game::GameStates S;
        storm::storage::BitVector state_space(restrictedStateSpace.size(), true);
        state_space.set(9, false);
        S.p1States = storm::storage::BitVector(restrictedStateSpace.size(), false);
        S.p1States.set(10, true); S.p1States.set(4, true);
        S.p2States = storm::storage::BitVector(enabledActions.size(), false);
        S.p2States.set(0, true);
        std::cout << "P2 attractors of S1=" << S.p1States << ", S2=" << S.p2States << std::endl;
        sw::game::BackwardTransitions backwardTransitions;
        game.initBackwardTransitions(backwardTransitions);
        sw::game::GameStates T = game.attractorsP2(S, backwardTransitions);
        std::cout << "S1=" << T.p1States << ", S2=" << T.p2States << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Bounded problem" << std::endl;
    std::cout << "BWmp=" << wmpGame->boundedProblem() << " | DBWmp=" << wmpGame->directBoundedProblem() << std::endl;

    storm::storage::Scheduler<double> scheduler(mdp->getNumberOfStates());
    sw::game::WindowParityGame<double> wpGame(*mdp, "priorities", restrictedStateSpace, enabledActions);
    std::cout << "BWpar=" << wpGame.boundedProblem() << " | DBWpar=" << wpGame.directBoundedProblem(scheduler) << std::endl;
    scheduler.printToStream(std::cout, mdp);

    // Bounded Window Objective
    std::cout << "Bounded Window Objectives: mean payoff (with game classification)" << std::endl;
    sw::BoundedWindow::BoundedWindowMeanPayoffObjective<double> boundedWindowMPObjectiveGame(*mdp, "weights");
    fwResult = sw::BoundedWindow::performMaxProb(boundedWindowMPObjectiveGame);
    std::cout << "Pr(BWmp) = [";
    for (uint_fast64_t state = 0; state < mdp->getNumberOfStates(); ++ state) {
        std::cout << "s" << state << "=" << fwResult.values[state] << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Bounded Window Objectives: parity" << std::endl;
    sw::BoundedWindow::BoundedWindowParityObjective<double> boundedWindowParityObjective(*mdp, "priorities");
    fwResult = sw::BoundedWindow::performMaxProb(boundedWindowParityObjective);
    std::cout << "Pr(BWpar) = [";
    for (uint_fast64_t state = 0; state < mdp->getNumberOfStates(); ++ state) {
        std::cout << "s" << state << "=" << fwResult.values[state] << ", ";
    }
    std::cout << "]" << std::endl;

}


void graphVizExample(){
    std::string prismModelPath = STORM_TEST_RESOURCES_DIR "/mdp/BndGWMP.prism";
    storm::storage::SymbolicModelDescription modelDescription = storm::parser::PrismParser::parse(prismModelPath);
    storm::prism::Program program = modelDescription.preprocess().asPrismProgram();
    storm::builder::BuilderOptions options = storm::builder::BuilderOptions(true, true);
    std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(program, options).build();

    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();

    storm::storage::SparseMatrix<double> matrix = mdp->getTransitionMatrix();
    storm::models::sparse::StandardRewardModel<double> rewardModel = model->getRewardModel("weights");
    std::vector<double> rewardVector = rewardModel.getStateActionRewardVector();

    sw::util::graphviz::GraphVizBuilder::mdpGraphExport(matrix, rewardVector);
}

void predecessorListExample() {
    std::string prismModelPath = STORM_SOURCE_DIR "/src/stochastic-windows/util/graphviz-examples/window_mp_par.prism";
    //std::string prismModelPath = STORM_TEST_RESOURCES_DIR "/mdp/sw_simple_example.prism";
    storm::storage::SymbolicModelDescription modelDescription = storm::parser::PrismParser::parse(prismModelPath);
    storm::prism::Program program = modelDescription.preprocess().asPrismProgram();
    storm::builder::BuilderOptions options = storm::builder::BuilderOptions(true, true);
    std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(program, options).build();

    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();
    storm::storage::MaximalEndComponentDecomposition<double> mecDecomposition(*mdp);

    mdp->printModelInformationToStream(std::cout);
    std::cout << mdp->getTransitionMatrix() << std::endl;
    storm::storage::BitVector restrictedStateSpace(mdp->getNumberOfStates(), true);
    storm::storage::BitVector enabledActions(mdp->getNumberOfChoices(), true);
    sw::storage::PredecessorsSquaredLinkedList<double> predList(mdp->getTransitionMatrix(), restrictedStateSpace, enabledActions);
    std::cout << predList << std::endl;
    predList.disableAction(7);
    predList.disableAction(2);
    std::cout << predList << std::endl;
}

int main(const int argc, const char** argv){

    // storm::utility::setUp();
    storm::cli::printHeader("Stochastic Windows (Storm backend)", argc, argv);
    initializeSettings();

    // mecDecompositionPrintExamples();
    // graphVizExample();
    windowExamples();
    schedulersExamples();
    // predecessorListExample();

    return 0;
}
