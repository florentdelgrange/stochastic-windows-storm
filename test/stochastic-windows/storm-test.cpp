//
// Created by Florent Delgrange on 2019-08-08.
//
#include <storm/settings/SettingsManager.h>
#include "gtest/gtest.h"

int main(int argc, char **argv) {
    storm::settings::initializeAll("Stochastic Windows Testing Suite", "test");
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
