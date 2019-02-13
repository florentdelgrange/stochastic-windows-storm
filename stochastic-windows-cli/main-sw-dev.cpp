//
// Created by Florent Delgrange on 25/10/2018.
//

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
#include <stochastic-windows/fixedwindow/MaximalEndComponentDecompositionUnfolding.h>
#include <stochastic-windows/fixedwindow/MeanPayoff.h>
#include <stochastic-windows/directfixedwindow/DirectFixedWindowObjective.h>
#include <stochastic-windows/game/WindowGame.h>
#include <stochastic-windows/game/PredecessorsSquaredLinkedList.h>
#include <stochastic-windows/fixedwindow/MaximalEndComponentDecompositionWindowGame.h>
#include <stochastic-windows/fixedwindow/MaximalEndComponentClassifier.h>
#include <stochastic-windows/fixedwindow/FixedWindowObjective.h>
#include <stochastic-windows/game/TotalPayoffGame.h>

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

    // storm::settings::mutableManager().printHelpForModule("minmax");
    storm::settings::mutableManager().setFromString("--minmax:method vi");
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


void windowExamples(){
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
    // std::cout << mdp->getTransitionMatrix().transpose() << std::endl;
    // std::cout << mdp->getTransitionMatrix().transpose(true) << std::endl;

    std::cout << mecDecomposition << std::endl;

    // maximum window size is 3
    // Mean Payoff
    sw::storage::MaximalEndComponentDecompositionUnfoldingMeanPayoff<double> unfoldingMp(*mdp, "weights", 3);
    std::cout << "unfolded matrices Mean Payoff: " << endl;
    for (uint_fast64_t k = 0; k < mecDecomposition.size(); ++ k) {
        unfoldingMp.printToStream(std::cout, k);
        std::shared_ptr<storm::models::sparse::Mdp<double>> new_mdp = unfoldingMp.unfoldingAsMDP(k);
        new_mdp->printModelInformationToStream(std::cout);
    }
    // Parity
    sw::storage::MaximalEndComponentDecompositionUnfoldingParity<double> unfoldingPar(*mdp, "priorities", 3);
    std::cout << "unfolded matrices Parity: " << endl;
    for (uint_fast64_t k = 0; k < mecDecomposition.size(); ++ k) {
        unfoldingPar.printToStream(std::cout, k);
        std::shared_ptr<storm::models::sparse::Mdp<double>> new_mdp = unfoldingPar.unfoldingAsMDP(k);
        new_mdp->printModelInformationToStream(std::cout);
    }

    // DirectFixed MP
    sw::DirectFixedWindow::DirectFixedWindowMeanPayoffObjective<double> dfwMpObjective(*mdp, "weights", 3);
    storm::storage::BitVector phiStates(mdp->getNumberOfStates(), false);
    phiStates.set(0, true); phiStates.set(5, true);
    std::unique_ptr<sw::DirectFixedWindow::WindowUnfolding<double>> unfoldingDirectFixedMP = dfwMpObjective.performUnfolding(phiStates);
    std::vector<double> result = sw::DirectFixedWindow::performMaxProb<double>(phiStates, dfwMpObjective);
    std::cout << "Pr(DFWmp) = [";
    for (auto state: phiStates) {
        std::cout << "s" << state << "=" << result[state] << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "result from s0 = " << sw::DirectFixedWindow::performMaxProb<double>(0, dfwMpObjective) << std::endl;
    // DirectFixed Par
    sw::DirectFixedWindow::DirectFixedWindowParityObjective<double> dfwParObjective(*mdp, "priorities", 3);
    phiStates = storm::storage::BitVector(mdp->getNumberOfStates(), true);
    std::unique_ptr<sw::DirectFixedWindow::WindowUnfolding<double>> unfoldingDirectFixedPar = dfwParObjective.performUnfolding(phiStates);
    result = sw::DirectFixedWindow::performMaxProb<double>(phiStates, dfwParObjective);
    std::cout << "Pr(DFWpar) = [";
    for (auto state: phiStates) {
        std::cout << "s" << state << "=" << result[state] << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Pr_s5(DFWpar) = " << sw::DirectFixedWindow::performMaxProb<double>(5, dfwParObjective) << std::endl;
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
    result = sw::FixedWindow::performMaxProb(fixedWindowMPObjectiveGame);
    std::cout << "Pr(FWmp) = [";
    for (uint_fast64_t state = 0; state < mdp->getNumberOfStates(); ++ state) {
        std::cout << "s" << state << "=" << result[state] << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Fixed Window Objectives: mean payoff (with unfolding based classification)" << std::endl;
    sw::FixedWindow::FixedWindowMeanPayoffObjective<double> fixedWindowMPObjectiveUnfolding(*mdp, "weights", 3, false);
    result = sw::FixedWindow::performMaxProb(fixedWindowMPObjectiveUnfolding);
    std::cout << "Pr(FWmp) = [";
    for (uint_fast64_t state = 0; state < mdp->getNumberOfStates(); ++ state) {
        std::cout << "s" << state << "=" << result[state] << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Fixed Window Objectives: parity" << std::endl;
    sw::FixedWindow::FixedWindowParityObjective<double> fixedWindowParityObjective(*mdp, "priorities", 3);
    result = sw::FixedWindow::performMaxProb(fixedWindowParityObjective);
    std::cout << "Pr(FWpar) = [";
    for (uint_fast64_t state = 0; state < mdp->getNumberOfStates(); ++ state) {
        std::cout << "s" << state << "=" << result[state] << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;

    sw::game::TotalPayoffGame<double> game(*mdp, "weights", restrictedStateSpace, enabledActions);
    std::cout << "actions of s_7 = ";
    for (auto s_prime: sw::game::TotalPayoffGame<double>::successorsP1(7, mdp->getTransitionMatrix(), enabledActions)) {
        std::cout << s_prime << ",";
    }
    std::cout << std::endl;
    storm::storage::BitVector newEnabledActions(mdp->getNumberOfChoices(), true);
    newEnabledActions.set(10, false);
    std::cout << "actions \\ (a10)  of s_7 = ";
    for (auto s_prime: sw::game::TotalPayoffGame<double>::successorsP1(7, mdp->getTransitionMatrix(), newEnabledActions)) {
        std::cout << s_prime << ",";
    }
    std::cout << std::endl;
    newEnabledActions.set(10, true);
    newEnabledActions.set(11, false);
    std::cout << "actions \\ (a11)  of s_7 = ";
    for (auto s_prime: sw::game::TotalPayoffGame<double>::successorsP1(7, mdp->getTransitionMatrix(), newEnabledActions)) {
        std::cout << s_prime << ",";
    }
    std::cout << std::endl;
    std::cout << "successors of action 6 = ";
    for (auto s_prime: sw::game::TotalPayoffGame<double>::successorsP2(6, mdp->getTransitionMatrix(), newEnabledActions)) {
        std::cout << s_prime << ",";
    }
    std::cout << std::endl;
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
    // predecessorListExample();

    return 0;
}
