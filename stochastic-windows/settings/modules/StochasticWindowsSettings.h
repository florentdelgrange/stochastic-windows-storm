//
// Created by florentdelgrange on 30/09/19.
//

#include "storm-config.h"
#include "storm/settings/modules/ModuleSettings.h"
#include "storm/settings/SettingsManager.h"
#include "storm/settings/SettingMemento.h"
#include "storm/settings/Option.h"
#include "storm/settings/OptionBuilder.h"
#include "storm/settings/ArgumentBuilder.h"
#include <stochastic-windows/boundedwindow/BoundedWindowObjective.h>

#ifndef STORM_STOCHASTICWINDOWSSETTINGS_H
#define STORM_STOCHASTICWINDOWSSETTINGS_H

namespace storm {
    namespace settings {
        namespace modules {
            class StochasticWindowsSettings : public ModuleSettings {
            public:
                enum class WindowObjective {DirectFixedWindowMeanPayoffObjective, DirectFixedWindowParityObjective, FixedWindowMeanPayoffObjective, FixedWindowParityObjective, BoundedWindowMeanPayoffObjective, BoundedWindowParityObjective};
                enum class ClassicalObjective {MeanPayoff, Parity};

                StochasticWindowsSettings();

                bool isExportSchedulerToDotFileSet() const;
                std::string getExportSchedulerDotFileName() const;
                bool isSchedulerLabelsSet() const;

                WindowObjective getWindowObjective() const;
                ClassicalObjective getClassicalObjective() const;

                std::string getRewardModelName() const;

                sw::BoundedWindow::ClassificationMethod getClassificationMethod() const;

                uint_fast64_t getMaximalWindowSize() const;

                bool check() const override;
                void finalize() override;

                // The name of the module.
                static const std::string moduleName;
            private:
                static const std::string objectiveOptionName;
                static const std::string windowSize;
                static const std::string exportDotSchedulerOptionName;
                static const std::string classificationMethodOptionName;
                static const std::string schedulerLabels;
                static const std::string rewardModelOptionName;
            };

        }
    }
}


#endif //STORM_STOCHASTICWINDOWSSETTINGS_H
