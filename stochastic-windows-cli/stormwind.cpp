//
// Created by Florent Delgrange on 25/10/2018.
//

#include "storm-config.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/storage/SymbolicModelDescription.h"
#include <iostream>

#include "storm/utility/initialize.h"

#include "stochastic-windows/settings/modules/StochasticWindowsSettings.h"
#include "storm/settings/modules/ModuleSettings.h"
#include "storm/settings/modules/GeneralSettings.h"
#include "storm/settings/modules/CoreSettings.h"
#include "storm/settings/modules/IOSettings.h"
#include "storm/settings/modules/DebugSettings.h"
#include "storm/settings/modules/GmmxxEquationSolverSettings.h"
#include "storm/settings/modules/NativeEquationSolverSettings.h"
#include "storm/settings/modules/EliminationSettings.h"
#include "storm/settings/modules/MinMaxEquationSolverSettings.h"
#include "storm/settings/modules/BisimulationSettings.h"
#include "storm/settings/modules/GlpkSettings.h"
#include "storm/settings/modules/ExplorationSettings.h"
#include "storm/settings/modules/ResourceSettings.h"
#include "storm/settings/modules/BuildSettings.h"
#include "storm/settings/modules/JitBuilderSettings.h"
#include "storm/settings/modules/TopologicalEquationSolverSettings.h"
#include "storm/settings/modules/MultiplierSettings.h"
#include "storm/api/export.h"

#include <storm/environment/solver/MinMaxSolverEnvironment.h>

#include <storm/storage/sparse/ModelComponents.h>
#include <storm/models/sparse/StateLabeling.h>
#include <storm/models/sparse/StandardRewardModel.h>

#include "storm-cli-utilities/cli.h"
#include "storm-cli-utilities/model-handling.h"

#include "stochastic-windows/directfixedwindow/DirectFixedWindowObjective.h"
#include "stochastic-windows/fixedwindow/FixedWindowObjective.h"
#include "stochastic-windows/util/Graphviz.h"

namespace sw {
    namespace cli {
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
            storm::settings::mutableManager().setName("StormWind", "stochastic-windows");
            storm::settings::addModule<storm::settings::modules::StochasticWindowsSettings>();
            storm::settings::addModule<storm::settings::modules::GeneralSettings>();
            storm::settings::addModule<storm::settings::modules::IOSettings>();
            storm::settings::addModule<storm::settings::modules::CoreSettings>();
            storm::settings::addModule<storm::settings::modules::DebugSettings>();
            storm::settings::addModule<storm::settings::modules::BuildSettings>();
            storm::settings::addModule<storm::settings::modules::BisimulationSettings>();
            storm::settings::addModule<storm::settings::modules::NativeEquationSolverSettings>();
            storm::settings::addModule<storm::settings::modules::EliminationSettings>();
            storm::settings::addModule<storm::settings::modules::MinMaxEquationSolverSettings>();
            storm::settings::addModule<storm::settings::modules::GlpkSettings>();
            storm::settings::addModule<storm::settings::modules::ExplorationSettings>();
            storm::settings::addModule<storm::settings::modules::TopologicalEquationSolverSettings>();
            storm::settings::addModule<storm::settings::modules::ResourceSettings>();
            storm::settings::addModule<storm::settings::modules::GmmxxEquationSolverSettings>();
            storm::settings::addModule<storm::settings::modules::MultiplierSettings>();
            storm::settings::addModule<storm::settings::modules::JitBuilderSettings>();
        }


        std::string getWindowObjectiveAsString() {
            const auto& swSettings = storm::settings::getModule<storm::settings::modules::StochasticWindowsSettings>();
            storm::settings::modules::StochasticWindowsSettings::WindowObjective windowObjective = swSettings.getWindowObjective();
            switch (windowObjective) {
                case storm::settings::modules::StochasticWindowsSettings::WindowObjective::DirectFixedWindowMeanPayoffObjective :
                    return "Direct fixed window mean payoff";
                case storm::settings::modules::StochasticWindowsSettings::WindowObjective::DirectFixedWindowParityObjective :
                    return "Direct fixed window parity";
                case storm::settings::modules::StochasticWindowsSettings::WindowObjective::FixedWindowMeanPayoffObjective:
                    return "Fixed window mean payoff";
                case storm::settings::modules::StochasticWindowsSettings::WindowObjective::FixedWindowParityObjective:
                    return "Fixed window parity";
                case storm::settings::modules::StochasticWindowsSettings::WindowObjective::BoundedWindowMeanPayoffObjective:
                    return "Bounded window mean payoff";
                case storm::settings::modules::StochasticWindowsSettings::WindowObjective::BoundedWindowParityObjective:
                    return "Bounded window parity";
            }
        }

        std::string getClassificationMethodAsString() {
            const auto& swSettings = storm::settings::getModule<storm::settings::modules::StochasticWindowsSettings>();
            sw::BoundedWindow::ClassificationMethod classificationMethod = swSettings.getClassificationMethod();
            switch (classificationMethod) {
                case sw::BoundedWindow::ClassificationMethod::Unfolding : return "unfolding";
                case sw::BoundedWindow::ClassificationMethod::WindowGameWithBound :  return "window game with polynomial memory strategy";
                case sw::BoundedWindow::ClassificationMethod::MemorylessWindowGame : return "window game with memoryless strategy";
            }
        }

    template <storm::dd::DdType DdType, typename BuildValueType, typename VerificationValueType = BuildValueType>
    void processInputWithValueTypeAndDdlib(storm::cli::SymbolicInput const& input) {
            const auto& coreSettings = storm::settings::getModule<storm::settings::modules::CoreSettings>();

            // For several engines, no model building step is performed, but the verification is started right away.
            storm::settings::modules::CoreSettings::Engine engine = coreSettings.getEngine();
            // build the model

            std::shared_ptr<storm::models::ModelBase> model = storm::cli::buildPreprocessExportModelWithValueTypeAndDdlib<DdType, BuildValueType, VerificationValueType>(input, engine);
            std::shared_ptr<storm::models::sparse::Mdp<VerificationValueType>> mdp = model->as<storm::models::sparse::Mdp<VerificationValueType>>();

            const auto& swSettings = storm::settings::getModule<storm::settings::modules::StochasticWindowsSettings>();
            const auto& ioSettings = storm::settings::getModule<storm::settings::modules::IOSettings>();

            // change transitions reward models to state actions reward models
            std::unordered_map<std::string, storm::models::sparse::StandardRewardModel<VerificationValueType>> rewardModels = mdp->getRewardModels();
            for (auto const& rewardModelTuple : rewardModels) {
                std::string const& rewardModelName = rewardModelTuple.first;
                storm::models::sparse::StandardRewardModel<VerificationValueType> const& rewardModel = rewardModelTuple.second;
                if (rewardModel.hasTransitionRewards()) {
                    storm::models::sparse::StandardRewardModel<VerificationValueType> newRewardModel = rewardModel;
                    mdp->removeRewardModel(rewardModelName);
                    newRewardModel.reduceToStateBasedRewards(mdp->getTransitionMatrix());
                    mdp->addRewardModel(rewardModelName, newRewardModel);
                }
            }

            storm::settings::modules::StochasticWindowsSettings::WindowObjective windowObjective = swSettings.getWindowObjective();
            std::unique_ptr<sw::storage::ValuesAndScheduler<VerificationValueType>> result;
            bool produceScheduler = swSettings.isExportSchedulerToDotFileSet() or ioSettings.isExportSchedulerSet();
            auto const& minMaxSettings = storm::settings::getModule<storm::settings::modules::MinMaxEquationSolverSettings>();
            auto minMaxEquationSolvingTechnique = minMaxSettings.getMinMaxEquationSolvingMethod();
            if(produceScheduler and minMaxEquationSolvingTechnique != storm::solver::MinMaxMethod::PolicyIteration) {
                STORM_LOG_ERROR("Equation solving method Policy Iteration is required to produce a scheduler (with --minmax:method pi).");
            }
            assert(not produceScheduler or minMaxEquationSolvingTechnique == storm::solver::MinMaxMethod::PolicyIteration);
        sw::BoundedWindow::ClassificationMethod classificationMethod = swSettings.getClassificationMethod();
            std::cout << "Objective: " << getWindowObjectiveAsString() << std::endl;
            std::cout << "End component classification method: " << getClassificationMethodAsString() << std::endl;
            std::cout << std::endl;
            switch (windowObjective) {
                case storm::settings::modules::StochasticWindowsSettings::WindowObjective::DirectFixedWindowMeanPayoffObjective :
                case storm::settings::modules::StochasticWindowsSettings::WindowObjective::DirectFixedWindowParityObjective : {
                    std::unique_ptr<sw::DirectFixedWindow::DirectFixedWindowObjective<VerificationValueType>> objective =
                            windowObjective == storm::settings::modules::StochasticWindowsSettings::WindowObjective::DirectFixedWindowMeanPayoffObjective ?
                            std::unique_ptr<sw::DirectFixedWindow::DirectFixedWindowObjective<VerificationValueType>>(new sw::DirectFixedWindow::DirectFixedWindowMeanPayoffObjective<VerificationValueType>(*mdp, swSettings.getRewardModelName(), swSettings.getMaximalWindowSize())) :
                            std::unique_ptr<sw::DirectFixedWindow::DirectFixedWindowObjective<VerificationValueType>>(new sw::DirectFixedWindow::DirectFixedWindowParityObjective<VerificationValueType>(*mdp, swSettings.getRewardModelName(), swSettings.getMaximalWindowSize()));
                    result = std::unique_ptr<sw::storage::ValuesAndScheduler<VerificationValueType>>(
                            new sw::storage::ValuesAndScheduler<VerificationValueType>(
                                    sw::DirectFixedWindow::performMaxProb(mdp->getInitialStates(), *objective, produceScheduler, swSettings.isSchedulerLabelsSet())));
                    break;
                }
                case storm::settings::modules::StochasticWindowsSettings::WindowObjective::FixedWindowMeanPayoffObjective:
                case storm::settings::modules::StochasticWindowsSettings::WindowObjective::FixedWindowParityObjective: {
                    bool windowGameBasedClassificationMethod = classificationMethod == sw::BoundedWindow::ClassificationMethod::WindowGameWithBound;
                    if (classificationMethod == sw::BoundedWindow::ClassificationMethod::MemorylessWindowGame) {
                        STORM_LOG_THROW(true, storm::exceptions::IllegalArgumentException, "A Bounded Window Game classification method cannot be used for a Fixed Window Objective.");
                    }
                    std::unique_ptr<sw::FixedWindow::FixedWindowObjective<VerificationValueType>> objective =
                            windowObjective == storm::settings::modules::StochasticWindowsSettings::WindowObjective::FixedWindowMeanPayoffObjective ?
                            std::unique_ptr<sw::FixedWindow::FixedWindowObjective<VerificationValueType>>(new sw::FixedWindow::FixedWindowMeanPayoffObjective<VerificationValueType>(*mdp, swSettings.getRewardModelName(), swSettings.getMaximalWindowSize(), windowGameBasedClassificationMethod)) :
                            std::unique_ptr<sw::FixedWindow::FixedWindowObjective<VerificationValueType>>(new sw::FixedWindow::FixedWindowParityObjective<VerificationValueType>(*mdp, swSettings.getRewardModelName(), swSettings.getMaximalWindowSize()));

                    result = std::unique_ptr<sw::storage::ValuesAndScheduler<VerificationValueType>>(
                            new sw::storage::ValuesAndScheduler<VerificationValueType>(
                                    sw::FixedWindow::performMaxProb(*objective, produceScheduler, swSettings.isSchedulerLabelsSet())));
                    break;
                }
                case storm::settings::modules::StochasticWindowsSettings::WindowObjective::BoundedWindowMeanPayoffObjective:
                case storm::settings::modules::StochasticWindowsSettings::WindowObjective::BoundedWindowParityObjective: {
                    std::unique_ptr<sw::BoundedWindow::BoundedWindowObjective<VerificationValueType>> objective =
                            windowObjective == storm::settings::modules::StochasticWindowsSettings::WindowObjective::BoundedWindowMeanPayoffObjective ?
                            std::unique_ptr<sw::BoundedWindow::BoundedWindowObjective<VerificationValueType>>(new sw::BoundedWindow::BoundedWindowMeanPayoffObjective<VerificationValueType>(*mdp, swSettings.getRewardModelName(), classificationMethod)) :
                            std::unique_ptr<sw::BoundedWindow::BoundedWindowObjective<VerificationValueType>>(new sw::BoundedWindow::BoundedWindowParityObjective<VerificationValueType>(*mdp, swSettings.getRewardModelName(), classificationMethod));
                    result = std::unique_ptr<sw::storage::ValuesAndScheduler<VerificationValueType>>(
                            new sw::storage::ValuesAndScheduler<VerificationValueType>(
                                    sw::BoundedWindow::performMaxProb(*objective, produceScheduler, swSettings.isSchedulerLabelsSet())));
                    break;
                }
            }
            STORM_PRINT("Result (Pmax) from initial states:" << std::endl);
            for (uint_fast64_t state : mdp->getInitialStates()) {
                std::ostringstream stream;
                uint_fast64_t i = mdp->getLabelsOfState(state).size();
                for (std::string const& label : mdp->getLabelsOfState(state)) {
                    -- i;
                    std::cout << label << (i ? ", " : "");
                }
                mdp->getLabelsOfState(state).empty() ? (std::cout << "s" << state << " ") : std::cout;
                std::cout << ": " << result->values[state] << std::endl;
            }
            if (ioSettings.isExportSchedulerSet()) {
                storm::storage::Scheduler<VerificationValueType> const& scheduler = *result->scheduler;
                STORM_PRINT_AND_LOG("Exporting scheduler ... ")
                storm::api::exportScheduler(mdp->template as<storm::models::sparse::Model<VerificationValueType>>(), scheduler, ioSettings.getExportSchedulerFilename());
            }
            if (ioSettings.isExportDotSet()) {
                auto parityOrMeanPayoff = [&] (storm::settings::modules::StochasticWindowsSettings::ClassicalObjective objective) -> std::string {
                    return swSettings.getClassicalObjective() == objective ? swSettings.getRewardModelName() : "";
                };
                sw::util::graphviz::GraphVizBuilder::mdpGraphExport<VerificationValueType>(
                        *mdp,
                        parityOrMeanPayoff(storm::settings::modules::StochasticWindowsSettings::ClassicalObjective::Parity),
                        parityOrMeanPayoff(storm::settings::modules::StochasticWindowsSettings::ClassicalObjective::MeanPayoff),
                        ioSettings.getExportDotFilename(),
                        ""
                );
            }
            if (swSettings.isExportSchedulerToDotFileSet()) {
                sw::storage::SchedulerProductLabeling labeling;
                if (swSettings.isSchedulerLabelsSet()) {
                    if (swSettings.getClassicalObjective() == storm::settings::modules::StochasticWindowsSettings::ClassicalObjective::Parity) {
                        labeling.priorities = swSettings.getRewardModelName();
                    } else {
                        labeling.weights = swSettings.getRewardModelName();
                    }
                }
                sw::util::graphviz::GraphVizBuilder::schedulerExport<VerificationValueType>(*mdp, *result->scheduler, labeling, swSettings.getExportSchedulerDotFileName());
            }
        }

        template <typename ValueType>
        void processInputWithValueType(storm::cli::SymbolicInput const& input) {
            const auto& coreSettings = storm::settings::getModule<storm::settings::modules::CoreSettings>();
            const auto& generalSettings = storm::settings::getModule<storm::settings::modules::GeneralSettings>();
            const auto& bisimulationSettings = storm::settings::getModule<storm::settings::modules::BisimulationSettings>();

            if (coreSettings.getDdLibraryType() == storm::dd::DdType::CUDD && coreSettings.isDdLibraryTypeSetFromDefaultValue() && generalSettings.isExactSet()) {
                STORM_LOG_INFO("Switching to DD library sylvan to allow for rational arithmetic.");
                sw::cli::processInputWithValueTypeAndDdlib<storm::dd::DdType::Sylvan, storm::RationalNumber>(input);
            } else if (coreSettings.getDdLibraryType() == storm::dd::DdType::CUDD && coreSettings.isDdLibraryTypeSetFromDefaultValue() && std::is_same<ValueType, double>::value && generalSettings.isBisimulationSet() && bisimulationSettings.useExactArithmeticInDdBisimulation()) {
                STORM_LOG_INFO("Switching to DD library sylvan to allow for rational arithmetic.");
                sw::cli::processInputWithValueTypeAndDdlib<storm::dd::DdType::Sylvan, storm::RationalNumber, double>(input);
            } else if (coreSettings.getDdLibraryType() == storm::dd::DdType::CUDD) {
                sw::cli::processInputWithValueTypeAndDdlib<storm::dd::DdType::CUDD, double>(input);
            } else {
                STORM_LOG_ASSERT(coreSettings.getDdLibraryType() == storm::dd::DdType::Sylvan, "Unknown DD library.");
                sw::cli::processInputWithValueTypeAndDdlib<storm::dd::DdType::Sylvan, ValueType>(input);
            }
        }

        void processOptions() {
            // Start by setting some urgent options (log levels, resources, etc.)
            storm::cli::setUrgentOptions();

            // Parse and preprocess symbolic input (PRISM, JANI, properties, etc.)
            storm::cli::SymbolicInput symbolicInput = storm::cli::parseAndPreprocessSymbolicInput();

            const auto& generalSettings = storm::settings::getModule<storm::settings::modules::GeneralSettings>();
            if (generalSettings.isParametricSet()) {
                STORM_LOG_THROW(false, storm::exceptions::NotSupportedException, "No parameters are supported for window objectives.");
            } else if (generalSettings.isExactSet()) {
#ifdef STORM_HAVE_CARL
                sw::cli::processInputWithValueType<storm::RationalNumber>(symbolicInput);
#else
                STORM_LOG_THROW(false, storm::exceptions::NotSupportedException, "No exact numbers are supported in this build.");
#endif
            } else {
                sw::cli::processInputWithValueType<double>(symbolicInput);
            }
        }
    }
}

int main(const int argc, const char** argv){
    try {
        storm::utility::setUp();
        storm::cli::printHeader("StormWind, running on Storm", argc, argv);
        sw::cli::initializeSettings();
        storm::settings::mutableManager().setFromString("--buildfull");

        storm::utility::Stopwatch totalTimer(true);
        storm::cli::parseOptions(argc, argv);

        std::cout << "Equation solving method: " << sw::cli::minMaxMethodAsString() << std::endl;
        sw::cli::processOptions();

        totalTimer.stop();
        if (storm::settings::getModule<storm::settings::modules::ResourceSettings>().isPrintTimeAndMemorySet()) {
            storm::cli::printTimeAndMemoryStatistics(totalTimer.getTimeInMilliseconds());
        }

        storm::utility::cleanUp();

    } catch (storm::exceptions::BaseException const& exception) {
        STORM_LOG_ERROR("An exception caused StormWind to terminate. The message of the exception is: " << exception.what());
        return 1;
    } catch (std::exception const& exception) {
        STORM_LOG_ERROR("An unexpected exception occurred and caused StormWind to terminate. The message of this exception is: " << exception.what());
        return 2;
    }

    return 0;
}
