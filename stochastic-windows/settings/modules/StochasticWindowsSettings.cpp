//
// Created by florentdelgrange on 30/09/19.
//

#include "StochasticWindowsSettings.h"
namespace storm {
    namespace settings {
        namespace modules {

            const std::string StochasticWindowsSettings::moduleName = "sw";
            const std::string StochasticWindowsSettings::objectiveOptionName = "objective";
            const std::string StochasticWindowsSettings::windowSize = "windowsize";
            const std::string StochasticWindowsSettings::exportDotSchedulerOptionName = "exportdotscheduler";
            const std::string StochasticWindowsSettings::classificationMethodOptionName = "classificationmethod";
            const std::string StochasticWindowsSettings::schedulerLabels = "schedulerslabels";

            StochasticWindowsSettings::StochasticWindowsSettings() : ModuleSettings(moduleName) {
                std::vector<std::string> windowVariants = {"dfw", "fw", "bw"};
                std::vector<std::string> longRunObjectives = {"mp", "par"};
                std::vector<std::string> classificationMethods = {"default", "bwgame", "dfwgame", "unfolding"};
                this->addOption(storm::settings::OptionBuilder(moduleName, objectiveOptionName, false, "Sets which window objective to consider.").setIsRequired(true)
                    .addArgument(storm::settings::ArgumentBuilder::createStringArgument("window variant", "The window variant to consider.")
                    .addValidatorString(ArgumentValidatorFactory::createMultipleChoiceValidator(windowVariants)).build())
                    .addArgument(storm::settings::ArgumentBuilder::createStringArgument("long run objective", "The long run objective to be strengthened with the window mecanism.")
                    .addValidatorString(ArgumentValidatorFactory::createMultipleChoiceValidator(longRunObjectives))
                    .build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, windowSize, false, "Sets the maximal window size.").setIsAdvanced()
                    .addArgument(storm::settings::ArgumentBuilder::createUnsignedIntegerArgument("value", "The window size.")
                    .addValidatorUnsignedInteger(storm::settings::ArgumentValidatorFactory::createUnsignedGreaterEqualValidator(1))
                    .build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, exportDotSchedulerOptionName, false, "If given, a scheduler maximizing the probability of the window objective will be written to the specified file in the dot format.").setIsAdvanced()
                    .addArgument(storm::settings::ArgumentBuilder::createStringArgument("filename", "The name of the file to which the scheduler is to be written.")
                    .build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, classificationMethodOptionName, false, "If given, sets the maximal end component classification method.")
                    .addArgument(storm::settings::ArgumentBuilder::createStringArgument("name", "The name of the method to use during the maximal end component classification.")
                    .addValidatorString(ArgumentValidatorFactory::createMultipleChoiceValidator(classificationMethods)).setDefaultValueString("default").build()).build());
                this->addOption(storm::settings::OptionBuilder(moduleName, schedulerLabels, false, "Add labels to the synthesized scheduler.").build());
            }

            bool StochasticWindowsSettings::isSchedulerLabelsSet() const {
                return this->getOption(schedulerLabels).getHasOptionBeenSet();
            }

            StochasticWindowsSettings::WindowObjective StochasticWindowsSettings::getWindowObjective() const {
                std::string windowVariant = this->getOption(objectiveOptionName).getArgumentByName("window variant").getValueAsString();
                std::string longRunObjective = this->getOption(objectiveOptionName).getArgumentByName("long run objective").getValueAsString();
                if (windowVariant == "dfw") return longRunObjective == "mp" ? WindowObjective::DirectFixedWindowMeanPayoffObjective : WindowObjective::DirectFixedWindowParityObjective;
                else if (windowVariant == "fw") return longRunObjective == "mp" ? WindowObjective::FixedWindowMeanPayoffObjective : WindowObjective::FixedWindowParityObjective;
                else return longRunObjective == "mp" ? WindowObjective::BoundedWindowMeanPayoffObjective : WindowObjective::BoundedWindowParityObjective;
            }

            sw::BoundedWindow::ClassificationMethod StochasticWindowsSettings::getClassificationMethod() const {
                std::string method = this->getOption(classificationMethodOptionName).getArgumentByName("name").getValueAsString();
                std::string windowVariant = this->getOption(objectiveOptionName).getArgumentByName("window variant").getValueAsString();
                if (windowVariant == "default") {
                    if (windowVariant == "dfw") return sw::BoundedWindow::ClassificationMethod::Unfolding;
                    else if (windowVariant == "fw") return sw::BoundedWindow::ClassificationMethod::WindowGameWithBound;
                    else return sw::BoundedWindow::ClassificationMethod::MemorylessWindowGame;
                } else if (windowVariant == "bwgame") return sw::BoundedWindow::ClassificationMethod::MemorylessWindowGame;
                else if (windowVariant == "dfwgame") return sw::BoundedWindow::ClassificationMethod::WindowGameWithBound;
                else return sw::BoundedWindow::ClassificationMethod::Unfolding;
            }

            uint_fast64_t StochasticWindowsSettings::getMaximalWindowSize() const {
                return this->getOption(windowSize).getArgumentByName("value").getValueAsUnsignedInteger();
            }

            bool StochasticWindowsSettings::check() const {
                return true;
            }

            void StochasticWindowsSettings::finalize() {

            }
        }
    }
}
