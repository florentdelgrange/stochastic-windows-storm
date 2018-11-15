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
#include<iostream>
#include <stochastic-windows/StochasticWindows.h>


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

#include "storm/analysis/GraphConditions.h"

#include "storm-cli-utilities/cli.h"
#include "storm-cli-utilities/model-handling.h"

#include "storm/api/storm.h"


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
    storm::settings::addModule<storm::settings::modules::GmmxxEquationSolverSettings>();
    storm::settings::addModule<storm::settings::modules::EigenEquationSolverSettings>();
    storm::settings::addModule<storm::settings::modules::NativeEquationSolverSettings>();
    storm::settings::addModule<storm::settings::modules::EliminationSettings>();
    storm::settings::addModule<storm::settings::modules::MinMaxEquationSolverSettings>();
    storm::settings::addModule<storm::settings::modules::GameSolverSettings>();
    storm::settings::addModule<storm::settings::modules::BisimulationSettings>();
    storm::settings::addModule<storm::settings::modules::GlpkSettings>();
    storm::settings::addModule<storm::settings::modules::ExplorationSettings>();
    storm::settings::addModule<storm::settings::modules::ResourceSettings>();
    storm::settings::addModule<storm::settings::modules::JitBuilderSettings>();
}


void mecDecompositionPrintTests() {

        std::string prismModelPath = STORM_TEST_RESOURCES_DIR "/mdp/sensors.prism";
        storm::storage::SymbolicModelDescription modelDescription = storm::parser::PrismParser::parse(prismModelPath);
        storm::prism::Program program = modelDescription.preprocess().asPrismProgram();
        storm::builder::BuilderOptions options = storm::builder::BuilderOptions(true, true);
        std::shared_ptr<storm::models::sparse::Model<double>> model = storm::builder::ExplicitModelBuilder<double>(program, options).build();

        std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();

        mdp->printModelInformationToStream(std::cout);
        storm::models::sparse::StandardRewardModel<double> rewardModel = model->getRewardModel("energy");

        storm::storage::MaximalEndComponentDecomposition<double> mecDecomposition(*mdp);

        std::cout << "MDP 1: Transition Matrix" << std::endl;
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
        // Note that each row of the transition matrix corresponds to an action in a given state.
        for (std::vector<double>::const_iterator i = rewardVector.begin(); i != rewardVector.end(); ++i)
            std::cout << *i << ' ';

        // create a vector of size |MECs|
        // std::vector<storm::storage::SparseMatrix<double>> matrices(mecDecomposition.size());

        storm::storage::SparseMatrix<double> originalMatrix = mdp->getTransitionMatrix();
        for(std::vector<storm::storage::MaximalEndComponent>::const_iterator mec = mecDecomposition.begin();
                mec != mecDecomposition.end(); ++mec){
            std::cout << "Mec -> " << *mec << std::endl;
            storm::storage::MaximalEndComponent::set_type stateSet = mec->getStateSet();
            for (storm::storage::MaximalEndComponent::set_type::const_iterator state = stateSet.begin();
                    state != stateSet.end(); ++state){
                std::cout << *state << ": ";
                storm::storage::MaximalEndComponent::set_type actionSet = mec->getChoicesForState(*state);
                for (storm::storage::MaximalEndComponent::set_type::const_iterator action = actionSet.begin();
                     action != actionSet.end(); ++action) {
                    std::cout << *action << ", ";
                }
            }
            std::cout << std::endl;
        }


};


int main(const int argc, const char** argv){

    storm::utility::setUp();
    storm::cli::printHeader("Stochastic Windows (Storm backend)", argc, argv);
    initializeSettings();

    mecDecompositionPrintTests();


    return 0;
}
