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

#include <storm/environment/solver/MinMaxSolverEnvironment.h>

#include <storm/storage/sparse/ModelComponents.h>
#include <storm/models/sparse/StateLabeling.h>

#include "storm-cli-utilities/cli.h"
#include "storm-cli-utilities/model-handling.h"

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

        template <storm::dd::DdType DdType, typename BuildValueType, typename VerificationValueType = BuildValueType>
        void processInputWithValueTypeAndDdlib(storm::cli::SymbolicInput const& input) {
            const auto& coreSettings = storm::settings::getModule<storm::settings::modules::CoreSettings>();

            // For several engines, no model building step is performed, but the verification is started right away.
            storm::settings::modules::CoreSettings::Engine engine = coreSettings.getEngine();

            std::shared_ptr<storm::models::ModelBase> model = storm::cli::buildPreprocessExportModelWithValueTypeAndDdlib<DdType, BuildValueType, VerificationValueType>(input, engine);
            std::shared_ptr<storm::models::sparse::Mdp<double>> mdp = model->as<storm::models::sparse::Mdp<double>>();

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
#ifdef STORM_HAVE_CARL
                sw::cli::processInputWithValueType<storm::RationalFunction>(symbolicInput);
#else
                STORM_LOG_THROW(false, storm::exceptions::NotSupportedException, "No parameters are supported in this build.");
#endif
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
        storm::cli::printHeader("StormWind, with Storm", argc, argv);
        sw::cli::initializeSettings();

        storm::utility::Stopwatch totalTimer(true);
        storm::cli::parseOptions(argc, argv);

        sw::cli::processOptions();
        std::cout << "Equation solving method: " << sw::cli::minMaxMethodAsString() << std::endl;

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
